use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_rfc3339(secs)
}

/// Second-precision RFC3339 formatter. Civil-from-days per Howard Hinnant.
pub fn format_rfc3339(mut secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    secs %= 86_400;
    let hour = (secs / 3600) as u32;
    let minute = ((secs / 60) % 60) as u32;
    let second = (secs % 60) as u32;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = y + if m <= 2 { 1 } else { 0 };
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Parse `YYYY-MM-DDTHH:MM:SSZ` into unix-epoch seconds. Accepts only the exact
/// shape palazzo itself produces (second-precision, Z-suffixed). Returns `None`
/// on shape mismatch or out-of-range fields.
pub fn parse_rfc3339(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return None;
    }
    fn num(slice: &[u8]) -> Option<i64> {
        std::str::from_utf8(slice).ok()?.parse().ok()
    }
    let y = num(&b[0..4])?;
    let m = num(&b[5..7])? as u32;
    let d = num(&b[8..10])? as u32;
    let h = num(&b[11..13])? as u32;
    let mi = num(&b[14..16])? as u32;
    let se = num(&b[17..19])? as u32;
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) || h >= 24 || mi >= 60 || se >= 60
    {
        return None;
    }
    // Howard Hinnant's days_from_civil — inverse of the civil-from-days in format_rfc3339.
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mi_shift = i64::from(if m > 2 { m - 3 } else { m + 9 });
    let doy = (153 * mi_shift + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h as i64 * 3600 + mi as i64 * 60 + se as i64)
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
            if leap { 29 } else { 28 }
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{format_rfc3339, parse_rfc3339};

    #[test]
    fn epoch_formats() {
        assert_eq!(format_rfc3339(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn known_date() {
        // 2026-04-20T08:37:19Z → 1776674239
        assert_eq!(format_rfc3339(1_776_674_239), "2026-04-20T08:37:19Z");
    }

    #[test]
    fn leap_day() {
        // 2024-02-29T12:34:56Z → 1709210096
        assert_eq!(format_rfc3339(1_709_210_096), "2024-02-29T12:34:56Z");
    }

    #[test]
    fn roundtrip_epoch() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
    }

    #[test]
    fn roundtrip_known_date() {
        assert_eq!(parse_rfc3339("2026-04-20T08:37:19Z"), Some(1_776_674_239));
    }

    #[test]
    fn roundtrip_leap_day() {
        assert_eq!(parse_rfc3339("2024-02-29T12:34:56Z"), Some(1_709_210_096));
    }

    #[test]
    fn roundtrip_many() {
        for s in [
            0_u64,
            86_400,
            1_000_000_000,
            1_700_000_000,
            1_776_674_239,
            2_000_000_000,
        ] {
            let formatted = format_rfc3339(s);
            let parsed = parse_rfc3339(&formatted).unwrap();
            assert_eq!(parsed as u64, s, "roundtrip failed for {s} → {formatted}");
        }
    }

    #[test]
    fn bad_shape() {
        assert!(parse_rfc3339("2026-04-20 08:37:19Z").is_none());
        assert!(parse_rfc3339("2026-04-20T08:37:19+00:00").is_none());
        assert!(parse_rfc3339("not a date").is_none());
    }

    #[test]
    fn out_of_range() {
        assert!(parse_rfc3339("2026-13-01T00:00:00Z").is_none());
        assert!(parse_rfc3339("2026-04-32T00:00:00Z").is_none());
        assert!(parse_rfc3339("2026-04-20T25:00:00Z").is_none());
    }

    #[test]
    fn impossible_calendar_days() {
        assert!(parse_rfc3339("2026-02-31T00:00:00Z").is_none());
        assert!(parse_rfc3339("2026-02-29T00:00:00Z").is_none()); // 2026 not a leap year
        assert!(parse_rfc3339("2024-02-29T00:00:00Z").is_some()); // 2024 is
        assert!(parse_rfc3339("2100-02-29T00:00:00Z").is_none()); // century non-leap
        assert!(parse_rfc3339("2000-02-29T00:00:00Z").is_some()); // 400-year leap
        assert!(parse_rfc3339("2026-04-31T00:00:00Z").is_none()); // April has 30
        assert!(parse_rfc3339("2026-04-00T00:00:00Z").is_none());
    }
}
