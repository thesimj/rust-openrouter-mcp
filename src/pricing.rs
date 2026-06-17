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
        "audio" | "audio_output" | "input_audio_cache" => per_m(v, "audio tokens"),
        "request" => format!("${}/request", trim_num(v)),
        "image" | "image_output" => format!("${}/image", trim_num(v)),
        "web_search" => format!("${}/call", trim_num(v)),
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
/// values. Returns `None` when nothing meaningful remains.
pub(crate) fn humanize_pricing(pricing: &Value) -> Option<Value> {
    let obj = pricing.as_object()?;
    let mut out = Map::new();
    for (k, val) in obj {
        if k == "discount" {
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
    if let Some(human) = obj.get("pricing").and_then(humanize_pricing) {
        if let Some(map) = obj.as_object_mut() {
            map.insert("pricing_human".to_string(), human);
        }
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
}
