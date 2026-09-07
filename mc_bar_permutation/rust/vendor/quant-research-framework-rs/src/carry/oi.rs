//! Open-interest stream loader (Rust mirror).

#![cfg(feature = "carry")]

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct OIRecord {
    pub time_s: i64,
    pub open_interest: f64,
    pub open_interest_usd: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct OIFrame {
    pub records: Vec<OIRecord>,
    pub expected_cadence_s: i64,
}

pub fn load_oi<P: AsRef<Path>>(
    path: P,
    expected_cadence_s: i64,
    cadence_tol_s: i64,
) -> Result<OIFrame, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path.as_ref())
        .map_err(|e| format!("load_oi: open {:?}: {}", path.as_ref(), e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("load_oi: headers: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let t_idx = headers
        .iter()
        .position(|h| h == "time")
        .ok_or_else(|| "load_oi: missing 'time'".to_string())?;
    let oi_idx = headers
        .iter()
        .position(|h| h == "open_interest")
        .ok_or_else(|| "load_oi: missing 'open_interest'".to_string())?;
    let oi_usd_idx = headers.iter().position(|h| h == "open_interest_usd");

    let mut records: Vec<OIRecord> = Vec::new();
    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("load_oi: row {}: {}", i, e))?;
        let t: i64 = rec.get(t_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_oi: row {} time: {}", i, e))?;
        let oi: f64 = rec.get(oi_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_oi: row {} oi: {}", i, e))?;
        if oi.is_nan() {
            return Err(format!("load_oi: NaN open_interest at row {}", i));
        }
        let oi_usd: Option<f64> = oi_usd_idx.and_then(|j| {
            rec.get(j).unwrap_or("").parse().ok()
        });
        records.push(OIRecord { time_s: t, open_interest: oi, open_interest_usd: oi_usd });
    }
    records.sort_by_key(|r| r.time_s);

    if records.len() >= 2 {
        let mut worst = 0i64;
        let mut worst_idx = 0usize;
        for i in 1..records.len() {
            let d = records[i].time_s - records[i - 1].time_s;
            let off = (d - expected_cadence_s).abs();
            if off > worst {
                worst = off;
                worst_idx = i - 1;
            }
        }
        if worst > cadence_tol_s {
            let actual = records[worst_idx + 1].time_s - records[worst_idx].time_s;
            return Err(format!(
                "load_oi: cadence at row {} = {}s, expected {}s ± {}s",
                worst_idx, actual, expected_cadence_s, cadence_tol_s
            ));
        }
    }
    Ok(OIFrame { records, expected_cadence_s })
}

pub fn oi_at(frame: &OIFrame, t_s: i64) -> Option<OIRecord> {
    let mut last: Option<OIRecord> = None;
    for rec in &frame.records {
        if rec.time_s <= t_s {
            last = Some(*rec);
        } else {
            break;
        }
    }
    last
}
