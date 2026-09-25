//! Price formatting for the MCP `list_models` and `describe_model` tools.
//! OpenRouter reports prices as USD-per-unit decimal strings.
//! Negative values are sentinels (e.g. `openrouter/auto` uses `-1` = "varies").

use serde_json::{Map, Value};

use crate::openrouter::Model;

/// Trim a float to a compact decimal string (up to 8 places, no trailing zeros).
fn trim_num(v: f64) -> String {
    let s = format!("{v:.8}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Known video SKU families, as `(factor to dollars, unit)`: `cents_*` and
/// `second*` keys are quoted in cents, `duration_seconds` keys (bare or with a
/// `text_to_video_`/`image_to_video_` task prefix) in dollars. Unknown names
/// retain their names and raw rate.
fn video_unit(key: &str) -> Option<(f64, &'static str)> {
    let task_free = key
        .strip_prefix("text_to_video_")
        .or_else(|| key.strip_prefix("image_to_video_"))
        .unwrap_or(key);
    if key.starts_with("cents_per_megapixel_second") {
        Some((0.01, "/MP-s"))
    } else if task_free.starts_with("duration_seconds") {
        Some((1.0, "/s"))
    } else if key.starts_with("second_")
        || key == "second"
        || key.starts_with("cents_per_second")
        || key.starts_with("cents_per_video_output_second")
        || key == "per-video-second"
    {
        Some((0.01, "/s"))
    } else if key == "cents_per_image_input" {
        Some((0.01, "/input image"))
    } else if key == "minimum_cents_per_generation" {
        Some((0.01, " minimum/video"))
    } else if key.starts_with("video_tokens") || key == "video_token" {
        Some((1_000_000.0, "/M vid-tok"))
    } else if key == "generate" {
        Some((1.0, "/video"))
    } else {
        None
    }
}

/// Humanize one OpenRouter price (a USD-per-unit decimal string) by pricing key.
/// Per-token fields become "$X/M tokens"; video SKUs use their real unit
/// (per-second, cents-per-second, or per-1M video tokens); others get their
/// natural unit. Zero, negative (sentinel), non-finite, and unparseable values
/// return `None` so they are omitted as noise.
fn humanize_price(key: &str, raw: &str) -> Option<String> {
    let v: f64 = raw.parse().ok()?;
    if !v.is_finite() || v <= 0.0 {
        return None;
    }
    let per_m = |n: f64, unit: &str| format!("${}/M {unit}", trim_num(n * 1_000_000.0));
    Some(match key {
        "prompt" | "completion" | "input_cache_read" | "input_cache_write"
        | "internal_reasoning" | "image_token" => per_m(v, "tokens"),
        // Generic catalog prose and provider rates disagree on this field.
        // The dedicated image endpoint supplies the authoritative unit.
        "image_output" => format!("${}/unit (see image endpoint)", trim_num(v)),
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
        k => match video_unit(k) {
            Some((scale, unit)) => format!("${}{unit}", trim_num(v * scale)),
            None => format!("${}/unit (unit unknown)", trim_num(v)),
        },
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
/// `pricing` is an array of `{billable, unit, cost_usd}` lines with NUMERIC costs
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
            let rendered = match l.get("unit").and_then(Value::as_str) {
                Some("token") => format!("${}/M tokens", trim_num(cost * 1_000_000.0)),
                Some("image") => format!("${}/image", trim_num(cost)),
                Some("megapixel") => format!("${}/MP", trim_num(cost)),
                Some("request") => format!("${}/request", trim_num(cost)),
                Some(unit) => format!("${}/{unit}", trim_num(cost)),
                None => format!("${}/unit (unit unknown)", trim_num(cost)),
            };
            let variant = match l.get("variant") {
                None | Some(Value::Null) => String::new(),
                Some(Value::String(s)) => format!(" [{s}]"),
                Some(other) => format!(" [{other}]"),
            };
            Some(Value::String(format!("{billable}{variant}: {rendered}")))
        })
        .collect();
    if !human.is_empty()
        && let Some(map) = endpoint.as_object_mut()
    {
        map.insert("pricing_human".to_string(), Value::Array(human));
    }
}

/// Serialize a model list to JSON, attaching a `pricing_human` sibling to each
/// model for the `list_models` MCP tool.
pub(crate) fn models_to_json(models: &[Model]) -> serde_json::Result<Value> {
    let mut v = serde_json::to_value(models)?;
    if let Some(arr) = v.as_array_mut() {
        for m in arr {
            attach_pricing_human(m);
        }
    }
    Ok(v)
}

/// Whether `pricing` expresses no price at all: every price (a decimal string
/// as OpenRouter sends them, or a bare number) is zero, or the object is empty
/// or absent. The non-price members are ignored: `discount` (a fraction) and
/// `overrides` (a list of conditional rates). Audio-output chat models such as Lyria report 0
/// token prices while billing a flat fee per track, so zero here means "not
/// expressed in this object", not "free".
pub(crate) fn is_zero_priced(pricing: &Value) -> bool {
    let Some(map) = pricing.as_object() else {
        return true;
    };
    map.iter()
        .filter(|(key, _)| !matches!(key.as_str(), "discount" | "overrides"))
        .all(|(_, v)| match v {
            Value::String(s) => s.trim().parse::<f64>().is_ok_and(|n| n == 0.0),
            Value::Number(n) => n.as_f64().is_some_and(|n| n == 0.0),
            _ => true,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_zero_priced_only_when_every_price_string_is_zero() {
        use serde_json::json;
        assert!(is_zero_priced(&json!({"prompt": "0", "completion": "0"})));
        assert!(is_zero_priced(
            &json!({"prompt": "0", "completion": "0.0", "discount": 0.25, "overrides": {}})
        ));
        // A bare number is a price too, even though OpenRouter sends strings.
        assert!(!is_zero_priced(&json!({"prompt": 0.0000025})));
        assert!(is_zero_priced(&json!({"prompt": 0, "completion": 0.0})));
        assert!(is_zero_priced(&json!({})));
        assert!(is_zero_priced(&Value::Null));
        assert!(!is_zero_priced(
            &json!({"prompt": "0", "audio_output": "0.000064"})
        ));
        // A sentinel ("varies") is a statement about price, not an absence of one.
        assert!(!is_zero_priced(&json!({"prompt": "-1"})));
        assert!(!is_zero_priced(&json!({"prompt": "n/a"})));
    }

    #[test]
    fn trim_num_drops_trailing_zeros_and_caps_precision() {
        assert_eq!(trim_num(1.0), "1");
        assert_eq!(trim_num(1.50), "1.5");
        assert_eq!(trim_num(0.12000000), "0.12");
        assert_eq!(trim_num(0.123456789), "0.12345679");
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
        // Generic image_output units are ambiguous. Explicit endpoint units
        // supply the authoritative rate; do not invent a per-image rate here.
        assert_eq!(
            humanize_price("image", "0.01").as_deref(),
            Some("$0.01/image")
        );
        assert_eq!(
            humanize_price("image_output", "0.0000119760479041916").as_deref(),
            Some("$0.00001198/unit (see image endpoint)")
        );
        // gemini-3.1-flash-image quotes image_output with no image_token beside it.
        assert_eq!(
            humanize_price("image_output", "0.00006").as_deref(),
            Some("$0.00006/unit (see image endpoint)")
        );
        // Video SKUs use their real units.
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

    /// Merged image endpoints carry numeric cost_usd lines; the human
    /// sibling renders them readably and skips zero/negative lines.
    #[test]
    fn attach_image_pricing_human_renders_numeric_cost_lines() {
        let mut ep = serde_json::json!({
            "provider_name": "OpenAI",
            "pricing": [
                {"billable": "output_image", "unit": "image", "cost_usd": 0.00004},
                {"billable": "input_text_tokens", "unit": "token", "cost_usd": 0.000005},
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

#[cfg(test)]
mod audit_regression {
    use super::*;
    #[test]
    fn explicit_image_units_override_billable_names_and_keep_variants() {
        let mut endpoint = serde_json::json!({"pricing": [
            {"billable":"output_image","unit":"token","cost_usd":0.00003,"variant":"high"},
            {"billable":"input_text","unit":"token","cost_usd":0.000005},
            {"billable":"input_image","unit":"megapixel","cost_usd":0.02},
            {"billable":"output_image","cost_usd":0.1}
        ]});
        attach_image_pricing_human(&mut endpoint);
        assert_eq!(
            endpoint["pricing_human"],
            serde_json::json!([
                "output_image [high]: $30/M tokens",
                "input_text: $5/M tokens",
                "input_image: $0.02/MP",
                "output_image: $0.1/unit (unit unknown)"
            ])
        );
    }
    #[test]
    fn video_rates_keep_megapixel_factors_and_unknown_units() {
        assert_eq!(
            humanize_price("cents_per_megapixel_second_precise", "7.5").as_deref(),
            Some("$0.075/MP-s")
        );
        assert_eq!(
            humanize_price("cents_per_megapixel_second_creative", "10.5").as_deref(),
            Some("$0.105/MP-s")
        );
        assert_eq!(
            humanize_price("unrecognized_second_unit", "2").as_deref(),
            Some("$2/unit (unit unknown)")
        );
    }

    /// SKU names seen live on `GET /videos/models` (2026-09-25): cents keys
    /// are cents, and the task-prefixed duration keys are dollars per second.
    #[test]
    fn video_rates_cover_the_live_sku_families() {
        for (key, raw, human) in [
            ("cents_per_video_output_second_1080p", "25", "$0.25/s"),
            ("cents_per_video_output_second_480p", "8", "$0.08/s"),
            ("cents_per_image_input", "1", "$0.01/input image"),
            ("minimum_cents_per_generation", "56", "$0.56 minimum/video"),
            ("text_to_video_duration_seconds_720p", "0.112", "$0.112/s"),
            ("image_to_video_duration_seconds_1080p", "0.084", "$0.084/s"),
            ("duration_seconds_with_audio_4k", "0.30", "$0.3/s"),
        ] {
            assert_eq!(humanize_price(key, raw).as_deref(), Some(human), "{key}");
        }
    }
    #[test]
    fn catalog_round_trip_preserves_one_hour_cache_write_pricing() {
        let model: Model = serde_json::from_value(serde_json::json!({
            "id":"anthropic/example", "pricing":{"input_cache_write_1h":"0.00001"}
        }))
        .unwrap();
        let output = models_to_json(&[model]).unwrap();
        assert_eq!(output[0]["pricing"]["input_cache_write_1h"], "0.00001");
        assert_eq!(
            output[0]["pricing_human"]["input_cache_write_1h"],
            "$10/M tokens"
        );
    }
}
