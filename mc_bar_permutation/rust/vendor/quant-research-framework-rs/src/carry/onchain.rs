//! On-chain stream loader with snapshot pinning (Rust mirror).
//!
//! The Rust port doesn't ship the SHA-256 helper (the Python side
//! pulls hashlib for free; the Rust side would need a `sha2` crate
//! and the metadata isn't load-bearing for parity since it's
//! ingestion-time provenance).  Consumers that need a snapshot ID
//! can add the dep later, the parity scripts compare values, not
//! provenance hashes.

#![cfg(feature = "carry")]

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct OnChainSnapshot {
    pub time_s: i64,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct OnChainFrame {
    pub records: Vec<OnChainSnapshot>,
    pub metric: String,
    pub snapshot_path: String,
}

pub fn load_onchain<P: AsRef<Path>>(path: P, metric: &str) -> Result<OnChainFrame, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path.as_ref())
        .map_err(|e| format!("load_onchain: open {:?}: {}", path.as_ref(), e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("load_onchain: headers: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let t_idx = headers
        .iter()
        .position(|h| h == "time")
        .ok_or_else(|| "load_onchain: missing 'time'".to_string())?;
    let m_idx = headers
        .iter()
        .position(|h| h == metric)
        .ok_or_else(|| format!("load_onchain: missing column '{}'", metric))?;

    let mut records: Vec<OnChainSnapshot> = Vec::new();
    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("load_onchain: row {}: {}", i, e))?;
        let t: i64 = rec.get(t_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_onchain: row {} time: {}", i, e))?;
        let v: f64 = rec.get(m_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_onchain: row {} value: {}", i, e))?;
        records.push(OnChainSnapshot { time_s: t, value: v });
    }
    records.sort_by_key(|r| r.time_s);
    Ok(OnChainFrame {
        records,
        metric: metric.to_string(),
        snapshot_path: path.as_ref().to_string_lossy().into_owned(),
    })
}

pub fn value_at(frame: &OnChainFrame, t_s: i64) -> Option<OnChainSnapshot> {
    let mut last: Option<OnChainSnapshot> = None;
    for rec in &frame.records {
        if rec.time_s <= t_s {
            last = Some(*rec);
        } else {
            break;
        }
    }
    last
}
