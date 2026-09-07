//! Funding-rate stream loader (Rust mirror).
//!
//! Loads a perpetual funding-rate series from a CSV with columns
//! `time, rate`.  Aligns to the venue's settlement boundary (8h for
//! every Binance / Bybit / OKX perp).  Mirrors `load_funding` from
//! `backtester.carry.funding`.

#![cfg(feature = "carry")]

use std::path::Path;

pub const FUNDING_INTERVAL_S: i64 = 8 * 3600;

#[derive(Debug, Clone, Copy)]
pub struct FundingEvent {
    pub time_s: i64,
    pub rate: f64,
}

#[derive(Debug, Clone)]
pub struct FundingFrame {
    pub events: Vec<FundingEvent>,
    pub venue: String,
    pub dup_count: usize,
}

impl FundingFrame {
    pub fn times(&self) -> Vec<i64> {
        self.events.iter().map(|e| e.time_s).collect()
    }
    pub fn rates(&self) -> Vec<f64> {
        self.events.iter().map(|e| e.rate).collect()
    }
}

/// Load a funding feed from CSV.  Verifies boundary alignment when
/// `strict_boundary` is true; rejects NaN rates; sorts ascending and
/// de-dupes on `time` (last-value-wins).
pub fn load_funding<P: AsRef<Path>>(
    path: P,
    venue: &str,
    strict_boundary: bool,
) -> Result<FundingFrame, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path.as_ref())
        .map_err(|e| format!("load_funding: open {:?}: {}", path.as_ref(), e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("load_funding: headers: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let t_idx = headers
        .iter()
        .position(|h| h == "time")
        .ok_or_else(|| "load_funding: missing 'time' column".to_string())?;
    let r_idx = headers
        .iter()
        .position(|h| h == "rate")
        .ok_or_else(|| "load_funding: missing 'rate' column".to_string())?;

    let mut raw: Vec<FundingEvent> = Vec::new();
    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("load_funding: row {}: {}", i, e))?;
        let t: i64 = rec
            .get(t_idx)
            .unwrap_or("")
            .parse()
            .map_err(|e| format!("load_funding: row {} time parse: {}", i, e))?;
        let r: f64 = rec
            .get(r_idx)
            .unwrap_or("")
            .parse()
            .map_err(|e| format!("load_funding: row {} rate parse: {}", i, e))?;
        if r.is_nan() {
            return Err(format!("load_funding: NaN rate at row {}", i));
        }
        raw.push(FundingEvent { time_s: t, rate: r });
    }

    // De-dup on time (last-value-wins to mirror Python's drop_duplicates(keep='last')).
    let mut by_time: std::collections::BTreeMap<i64, f64> =
        std::collections::BTreeMap::new();
    let n_raw = raw.len();
    for ev in raw {
        by_time.insert(ev.time_s, ev.rate);
    }
    let dup_count = n_raw.saturating_sub(by_time.len());
    let events: Vec<FundingEvent> = by_time
        .into_iter()
        .map(|(t, r)| FundingEvent { time_s: t, rate: r })
        .collect();

    if strict_boundary {
        for (i, ev) in events.iter().enumerate() {
            if ev.time_s.rem_euclid(FUNDING_INTERVAL_S) != 0 {
                return Err(format!(
                    "load_funding: row {} time={} not aligned to {}s {} boundary",
                    i, ev.time_s, FUNDING_INTERVAL_S, venue
                ));
            }
        }
    }

    Ok(FundingFrame {
        events,
        venue: venue.to_string(),
        dup_count,
    })
}

/// Smallest funding settlement timestamp `>= t_s`.
pub fn next_funding_time(t_s: i64, interval_s: i64) -> i64 {
    let rem = t_s.rem_euclid(interval_s);
    if rem == 0 {
        t_s
    } else {
        t_s + (interval_s - rem)
    }
}

/// Most-recent funding rate at-or-before `t_s`.  `None` if no event
/// exists at or before `t_s`.
pub fn rate_at(frame: &FundingFrame, t_s: i64) -> Option<f64> {
    let mut last: Option<f64> = None;
    for ev in &frame.events {
        if ev.time_s <= t_s {
            last = Some(ev.rate);
        } else {
            break;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_funding_time_aligned_returns_self() {
        assert_eq!(next_funding_time(1773072000, FUNDING_INTERVAL_S), 1773072000);
        assert_eq!(
            next_funding_time(1773072001, FUNDING_INTERVAL_S),
            1773072000 + FUNDING_INTERVAL_S
        );
    }

    #[test]
    fn rate_at_returns_most_recent() {
        let frame = FundingFrame {
            events: vec![
                FundingEvent { time_s: 0, rate: 1e-4 },
                FundingEvent { time_s: 28800, rate: 2e-4 },
                FundingEvent { time_s: 57600, rate: 3e-4 },
            ],
            venue: "binance_perp".to_string(),
            dup_count: 0,
        };
        assert_eq!(rate_at(&frame, -1), None);
        assert_eq!(rate_at(&frame, 0), Some(1e-4));
        assert_eq!(rate_at(&frame, 28799), Some(1e-4));
        assert_eq!(rate_at(&frame, 28800), Some(2e-4));
        assert_eq!(rate_at(&frame, 999999), Some(3e-4));
    }
}
