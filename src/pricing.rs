//! Shared price formatting for OpenRouter prices, used by both the CLI table
//! renderer and the MCP `list_models`/`describe_model` tools so the two never
//! diverge. OpenRouter reports prices as USD-per-unit decimal strings; negative
//! values are sentinels (e.g. `openrouter/auto` uses `-1` = "varies").

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::openrouter::Model;

/// Trim a float to a compact decimal string (up to 8 places, no trailing zeros).
pub(crate) fn trim_num(v: f64) -> String {
    let s = format!("{v:.8}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Format an OpenRouter per-token price string as USD per 1M tokens, with the
/// compact 4-decimal precision used by the table view. Returns "-" when
/// missing/unparseable/negative and "0" for zero.
pub(crate) fn per_million(price: &Option<String>) -> String {
    match price.as_deref().and_then(|s| s.parse::<f64>().ok()) {
        Some(0.0) => "0".to_string(),
        // Negative values are sentinels (e.g. openrouter/auto uses -1 = "varies").
        Some(p) if p < 0.0 => "-".to_string(),
        Some(p) if p.is_finite() => {
            let s = format!("{:.4}", p * 1_000_000.0);
            let s = s.trim_end_matches('0').trim_end_matches('.');
            format!("${s}")
        }
        _ => "-".to_string(),
    }
}

/// Render a list of prices as `$x<unit>` or `$min-max<unit>`.
pub(crate) fn range_str(vals: &[f64], unit: &str) -> String {
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        format!("${}{}", trim_num(min), unit)
    } else {
        format!("${}-{}{}", trim_num(min), trim_num(max), unit)
    }
}

/// Derive a concise price for a video model from its heterogeneous
/// `pricing_skus`: dollars-per-second, cents-per-second, or per-1M video tokens.
pub(crate) fn video_price(skus: &BTreeMap<String, String>) -> String {
    let collect = |pred: &dyn Fn(&str) -> bool| -> Vec<f64> {
        skus.iter()
            .filter(|(k, _)| pred(k))
            .filter_map(|(_, v)| v.parse::<f64>().ok())
            .collect()
    };
    let secs = collect(&|k| k.contains("duration_seconds"));
    if !secs.is_empty() {
        return range_str(&secs, "/s");
    }
    let cents = collect(&|k| k.contains("second"));
    if !cents.is_empty() {
        let dollars: Vec<f64> = cents.iter().map(|c| c / 100.0).collect();
        return range_str(&dollars, "/s");
    }
    let toks = collect(&|k| k.contains("token"));
    if !toks.is_empty() {
        let per_m: Vec<f64> = toks.iter().map(|t| t * 1_000_000.0).collect();
        return range_str(&per_m, "/M vid-tok");
    }
    "-".to_string()
}

/// Humanize one OpenRouter price (a USD-per-unit decimal string) by pricing key.
/// Per-token fields become "$X/M tokens"; video SKUs use their real unit
/// (per-second, cents-per-second, or per-1M video tokens); others get their
/// natural unit. Zero, negative (sentinel), non-finite, and unparseable values
/// return `None` so they are omitted as noise.
pub(crate) fn humanize_price(key: &str, raw: &str) -> Option<String> {
    let v: f64 = raw.parse().ok()?;
    if !v.is_finite() || v <= 0.0 {
        return None;
    }
    let per_m = |n: f64, unit: &str| format!("${}/M {unit}", trim_num(n * 1_000_000.0));
    Some(match key {
        "prompt" | "completion" | "input_cache_read" | "input_cache_write"
        | "internal_reasoning" | "image_token" => per_m(v, "tokens"),
        // Per *output token*, despite the name - verified against two providers.
        // x-ai/grok-imagine-image-quality reports image_output == image_token
        // (1.198e-5), and 8 images at 2K billed $0.64, i.e. ~6.7k tokens each.
        // google/gemini-3.1-flash-image reports 6e-5 with no image_token, and
        // its documented 1K price ($0.067) is ~1.1k tokens at that rate.
        // Rendered as "$/image" this read as $0.00006 per image.
        "image_output" => per_m(v, "output tokens"),
        "audio" | "audio_output" | "input_audio_cache" => per_m(v, "audio tokens"),
        "request" => format!("${}/request", trim_num(v)),
        // A flat per-image SKU that is not always what bills: grok-imagine
        // advertises $0.01/image here while charging ~$0.08 through image_token.
        "image" => format!("${}/image", trim_num(v)),
        "web_search" => format!("${}/call", trim_num(v)),
        // TTL siblings of the cache-write rate (`_1h`, and whatever follows) are
        // priced per token like the base key. Without this they fell to the `_`
        // catch-all and rendered as "$0.00002/unit" beside "$12.5/M tokens" - the
        // same unit shown two ways, 10^6 apart.
        k if k.starts_with("input_cache_write") => per_m(v, "tokens"),
        // Video SKUs: match the conventions in `video_price`.
        k if k.contains("duration_seconds") => format!("${}/s", trim_num(v)),
        k if k.contains("second") => format!("${}/s", trim_num(v / 100.0)), // cents -> dollars
        k if k.contains("token") => per_m(v, "vid-tok"),                    // video tokens, per 1M
        "generate" => format!("${}/video", trim_num(v)),
        _ => format!("${}/unit", trim_num(v)),
    })
}

/// Build a human-readable sibling for a pricing object: maps each price string
/// to its "$X/unit" form, skipping zeros/negatives, `discount`, and non-string
/// values. `overrides[]` entries render the same way: price keys (those also
/// present in the flat object) are humanized-or-dropped, condition keys pass
/// through verbatim, and priceless overrides are omitted. Returns `None` when
/// nothing meaningful remains.
pub(crate) fn humanize_pricing(pricing: &Value) -> Option<Value> {
    let obj = pricing.as_object()?;
    let mut out = Map::new();
    for (k, val) in obj {
        if k == "discount" {
            continue;
        }
        // Tiered/time-window pricing. Only a key that also exists in the parent
        // pricing object is a price (overrides override flat rates); everything
        // else is a condition, kept verbatim. Prices humanize_price rejects
        // (zero/sentinel/garbage) are dropped exactly like the flat path drops
        // them, and an override with no humanized price at all is noise.
        if k == "overrides" {
            if let Some(arr) = val.as_array() {
                let hum: Vec<Value> = arr
                    .iter()
                    .filter_map(|o| {
                        let ov = o.as_object()?;
                        let mut m = Map::new();
                        let mut priced = false;
                        for (ok, oval) in ov {
                            if ok == "discount" {
                                continue;
                            }
                            if obj.contains_key(ok) {
                                if let Some(h) = oval.as_str().and_then(|s| humanize_price(ok, s)) {
                                    m.insert(ok.clone(), Value::String(h));
                                    priced = true;
                                }
                            } else {
                                m.insert(ok.clone(), oval.clone());
                            }
                        }
                        (priced && !m.is_empty()).then_some(Value::Object(m))
                    })
                    .collect();
                if !hum.is_empty() {
                    out.insert(k.clone(), Value::Array(hum));
                }
            }
            continue;
        }
        if let Some(human) = val.as_str().and_then(|s| humanize_price(k, s)) {
            out.insert(k.clone(), Value::String(human));
        }
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

/// Attach a `pricing_human` sibling next to a `pricing` object in `obj`, in
/// place, when one can be built.
pub(crate) fn attach_pricing_human(obj: &mut Value) {
    if let Some(human) = obj.get("pricing").and_then(humanize_pricing)
        && let Some(map) = obj.as_object_mut()
    {
        map.insert("pricing_human".to_string(), human);
    }
}

/// Attach a `pricing_human` sibling to one merged image-endpoint object. Its
/// `pricing` is an array of `{billable, cost_usd}` lines with NUMERIC costs
/// (unlike the string-priced flat pricing objects), rendered as "billable: $X".
pub(crate) fn attach_image_pricing_human(endpoint: &mut Value) {
    let Some(lines) = endpoint.get("pricing").and_then(Value::as_array) else {
        return;
    };
    let human: Vec<Value> = lines
        .iter()
        .filter_map(|l| {
            let billable = l.get("billable")?.as_str()?;
            let cost = l.get("cost_usd")?.as_f64()?;
            if !(cost > 0.0 && cost.is_finite()) {
                return None;
            }
            // Same unit conventions as humanize_price: per-token rates scale
            // to $/M, per-image rates read per image, the rest say /unit.
            let rendered = if billable.contains("token") {
                format!("${}/M tokens", trim_num(cost * 1_000_000.0))
            } else if billable.contains("image") {
                format!("${}/image", trim_num(cost))
            } else {
                format!("${}/unit", trim_num(cost))
            };
            Some(Value::String(format!("{billable}: {rendered}")))
        })
        .collect();
    if !human.is_empty()
        && let Some(map) = endpoint.as_object_mut()
    {
        map.insert("pricing_human".to_string(), Value::Array(human));
    }
}

/// Serialize a model list to JSON, attaching a `pricing_human` sibling to each
/// model. Shared by the CLI `models` JSON output and the `list_models` MCP tool
/// so both render pricing identically.
pub(crate) fn models_to_json(models: &[Model]) -> Value {
    let mut v = serde_json::to_value(models).unwrap_or_else(|_| Value::Array(Vec::new()));
    if let Some(arr) = v.as_array_mut() {
        for m in arr {
            attach_pricing_human(m);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_million_formats_prices_and_sentinels() {
        assert_eq!(per_million(&Some("0".to_string())), "0");
        assert_eq!(per_million(&Some("-1".to_string())), "-");
        assert_eq!(per_million(&Some("0.00000075".to_string())), "$0.75");
        assert_eq!(per_million(&Some("0.00003".to_string())), "$30");
        assert_eq!(per_million(&Some("not-a-number".to_string())), "-");
        assert_eq!(per_million(&None), "-");
    }

    #[test]
    fn trim_num_drops_trailing_zeros_and_caps_precision() {
        assert_eq!(trim_num(1.0), "1");
        assert_eq!(trim_num(1.50), "1.5");
        assert_eq!(trim_num(0.12000000), "0.12");
        assert_eq!(trim_num(0.123456789), "0.12345679");
    }

    #[test]
    fn range_str_collapses_equal_bounds_and_renders_ranges() {
        assert_eq!(range_str(&[0.02], "/s"), "$0.02/s");
        assert_eq!(range_str(&[0.5, 0.5], "/s"), "$0.5/s");
        assert_eq!(range_str(&[0.02, 0.03], "/s"), "$0.02-0.03/s");
        assert_eq!(range_str(&[0.03, 0.02], "/s"), "$0.02-0.03/s");
    }

    #[test]
    fn video_price_prefers_seconds_then_cents_then_video_tokens() {
        let mut skus = BTreeMap::new();
        skus.insert("duration_seconds".to_string(), "0.12".to_string());
        skus.insert("video_tokens".to_string(), "0.01".to_string());
        assert_eq!(video_price(&skus), "$0.12/s");

        let mut skus = BTreeMap::new();
        skus.insert("second_with_audio".to_string(), "3".to_string());
        skus.insert("second_without_audio".to_string(), "2".to_string());
        assert_eq!(video_price(&skus), "$0.02-0.03/s");

        // Video tokens now normalize to per-1M for readability.
        let mut skus = BTreeMap::new();
        skus.insert("video_tokens".to_string(), "0.000007".to_string());
        assert_eq!(video_price(&skus), "$7/M vid-tok");

        assert_eq!(video_price(&BTreeMap::new()), "-");
    }

    #[test]
    fn humanize_price_units_skip_zero_and_negative_sentinel() {
        // Per-token text fields -> $X/M tokens.
        assert_eq!(
            humanize_price("prompt", "0.000005").as_deref(),
            Some("$5/M tokens")
        );
        assert_eq!(
            humanize_price("completion", "0.000025").as_deref(),
            Some("$25/M tokens")
        );
        assert_eq!(
            humanize_price("input_cache_read", "0.0000005").as_deref(),
            Some("$0.5/M tokens")
        );
        assert_eq!(
            humanize_price("input_cache_write", "0.0000125").as_deref(),
            Some("$12.5/M tokens")
        );
        assert_eq!(
            humanize_price("input_cache_write_1h", "0.00002").as_deref(),
            Some("$20/M tokens")
        );
        // `image` is a flat per-image SKU; `image_output` is per output token
        // despite the name, so it must not render in the same unit. The two
        // values below are the real ones from grok-imagine-image-quality, where
        // reading image_output as per-image understated the cost ~8x.
        assert_eq!(
            humanize_price("image", "0.01").as_deref(),
            Some("$0.01/image")
        );
        assert_eq!(
            humanize_price("image_output", "0.0000119760479041916").as_deref(),
            Some("$11.9760479/M output tokens")
        );
        // gemini-3.1-flash-image quotes image_output with no image_token beside it.
        assert_eq!(
            humanize_price("image_output", "0.00006").as_deref(),
            Some("$60/M output tokens")
        );
        // Video SKUs use their real units (matching video_price).
        assert_eq!(
            humanize_price("video_tokens", "0.000007").as_deref(),
            Some("$7/M vid-tok")
        );
        assert_eq!(
            humanize_price("duration_seconds", "0.12").as_deref(),
            Some("$0.12/s")
        );
        // `second_*` keys are cents-per-second -> dollars.
        assert_eq!(
            humanize_price("second_with_audio", "5").as_deref(),
            Some("$0.05/s")
        );
        assert_eq!(
            humanize_price("request", "0.01").as_deref(),
            Some("$0.01/request")
        );
        // Zero, negative sentinel, non-finite, and garbage are dropped.
        assert_eq!(humanize_price("prompt", "0"), None);
        assert_eq!(humanize_price("prompt", "-1"), None);
        assert_eq!(humanize_price("prompt", "NaN"), None);
        assert_eq!(humanize_price("prompt", "abc"), None);
    }

    #[test]
    fn humanize_pricing_skips_discount_and_zeros() {
        let p = serde_json::json!({"prompt": "0.000005", "completion": "0", "discount": 0.5});
        let human = humanize_pricing(&p).unwrap();
        assert_eq!(human["prompt"], "$5/M tokens");
        assert!(human.get("completion").is_none());
        assert!(human.get("discount").is_none());
    }

    /// Tiered/time-window overrides get their own humanized schedule: price
    /// strings render as "$X/M tokens", condition fields pass through verbatim.
    #[test]
    fn humanize_pricing_renders_overrides_schedule() {
        let p = serde_json::json!({
            "prompt": "0.000005",
            "overrides": [
                {"min_prompt_tokens": 200000, "prompt": "0.00001"},
                {"start_time": "18:30", "end_time": "23:30", "prompt": "0.0000025"}
            ]
        });
        let human = humanize_pricing(&p).unwrap();
        assert_eq!(human["prompt"], "$5/M tokens");
        let ov = human["overrides"].as_array().unwrap();
        assert_eq!(ov[0]["min_prompt_tokens"], 200000);
        assert_eq!(ov[0]["prompt"], "$10/M tokens");
        assert_eq!(ov[1]["start_time"], "18:30");
        assert_eq!(ov[1]["prompt"], "$2.5/M tokens");
    }

    /// Only keys that exist in the flat pricing object are prices: a numeric-
    /// string condition must never render as a dollar figure, sentinel/zero
    /// prices are dropped like the flat path drops them, and an override with
    /// no humanized price is omitted entirely.
    #[test]
    fn humanize_pricing_overrides_never_mislabel_conditions_or_leak_sentinels() {
        let p = serde_json::json!({
            "prompt": "0.000005",
            "overrides": [
                {"min_prompt_tokens": "200000", "hours": "18", "prompt": "0.00001"},
                {"prompt": "0", "completion": "-1"},
                {"condition": "peak_hours"}
            ]
        });
        let human = humanize_pricing(&p).unwrap();
        let ov = human["overrides"].as_array().unwrap();
        assert_eq!(ov.len(), 1, "priceless overrides are dropped: {ov:?}");
        assert_eq!(ov[0]["min_prompt_tokens"], "200000");
        assert_eq!(ov[0]["hours"], "18");
        assert_eq!(ov[0]["prompt"], "$10/M tokens");

        let junk_only = serde_json::json!({"overrides": [{"condition": "peak_hours"}]});
        assert!(humanize_pricing(&junk_only).is_none());
    }

    /// F11: merged image endpoints carry numeric cost_usd lines; the human
    /// sibling renders them readably and skips zero/negative lines.
    #[test]
    fn attach_image_pricing_human_renders_numeric_cost_lines() {
        let mut ep = serde_json::json!({
            "provider_name": "OpenAI",
            "pricing": [
                {"billable": "output_image", "cost_usd": 0.00004},
                {"billable": "input_text_tokens", "cost_usd": 0.000005},
                {"billable": "zeroed_tokens", "cost_usd": 0.0},
                {"billable": "weird", "cost_usd": -1.0}
            ]
        });
        attach_image_pricing_human(&mut ep);
        assert_eq!(
            ep["pricing_human"],
            serde_json::json!([
                "output_image: $0.00004/image",
                "input_text_tokens: $5/M tokens"
            ])
        );

        let mut no_pricing = serde_json::json!({"provider_name": "X"});
        attach_image_pricing_human(&mut no_pricing);
        assert!(no_pricing.get("pricing_human").is_none());
    }
}
