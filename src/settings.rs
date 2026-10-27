//! Numeric settings read from environment variables.

use std::str::FromStr;

/// The numeric environment variable `name`, or `None` when it is unset or
/// blank. A value that does not parse is reported on stderr and treated as
/// unset, so the documented default applies and the operator can see why.
pub(crate) fn env_number<T: FromStr>(name: &str) -> Option<T> {
    parse_number(name, std::env::var(name).ok().as_deref())
}

fn parse_number<T: FromStr>(name: &str, raw: Option<&str>) -> Option<T> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    let parsed = raw.parse().ok();
    if parsed.is_none() {
        eprintln!("warning: ignoring {name}={raw:?}: expected a whole number, using the default");
    }
    parsed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trimmed_whole_numbers_and_treats_blank_or_invalid_as_unset() {
        assert_eq!(parse_number::<u64>("X", None), None);
        assert_eq!(parse_number::<u64>("X", Some("  ")), None);
        assert_eq!(parse_number::<u64>("X", Some(" 12 ")), Some(12));
        assert_eq!(parse_number::<u64>("X", Some("20m")), None);
        assert_eq!(parse_number::<u64>("X", Some("-5")), None);
    }
}
