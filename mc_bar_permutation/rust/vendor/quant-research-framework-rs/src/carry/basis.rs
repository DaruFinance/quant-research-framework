//! Perp-vs-spot basis loader (Rust mirror).

#![cfg(feature = "carry")]

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct BasisRecord {
    pub time_s: i64,
    pub close_spot: f64,
    pub close_perp: f64,
    pub basis_bp: f64,
}

#[derive(Debug, Clone)]
pub struct BasisFrame {
    pub records: Vec<BasisRecord>,
    pub instrument_pair: String,
}

fn basis_bp(perp: f64, spot: f64) -> f64 {
    if spot == 0.0 {
        f64::NAN
    } else {
        (perp - spot) / spot * 1e4
    }
}

/// Load a basis CSV.  Required columns: `time, close_spot, close_perp`.
/// Optional `basis_bp` is recomputed and verified to agree with the
/// fresh calculation within `recompute_basis_tol_bp`.
pub fn load_basis<P: AsRef<Path>>(
    path: P,
    instrument_pair: &str,
    recompute_basis_tol_bp: f64,
) -> Result<BasisFrame, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path.as_ref())
        .map_err(|e| format!("load_basis: open {:?}: {}", path.as_ref(), e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("load_basis: headers: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let t_idx = headers
        .iter()
        .position(|h| h == "time")
        .ok_or_else(|| "load_basis: missing 'time'".to_string())?;
    let s_idx = headers
        .iter()
        .position(|h| h == "close_spot")
        .ok_or_else(|| "load_basis: missing 'close_spot'".to_string())?;
    let p_idx = headers
        .iter()
        .position(|h| h == "close_perp")
        .ok_or_else(|| "load_basis: missing 'close_perp'".to_string())?;
    let stored_bp_idx = headers.iter().position(|h| h == "basis_bp");

    let mut records: Vec<BasisRecord> = Vec::new();
    let mut worst_drift = 0.0f64;
    let mut worst_drift_row = 0usize;
    for (i, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("load_basis: row {}: {}", i, e))?;
        let t: i64 = rec.get(t_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_basis: row {} time: {}", i, e))?;
        let s: f64 = rec.get(s_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_basis: row {} spot: {}", i, e))?;
        let p: f64 = rec.get(p_idx).unwrap_or("").parse()
            .map_err(|e| format!("load_basis: row {} perp: {}", i, e))?;
        let fresh = basis_bp(p, s);
        if let Some(bp_i) = stored_bp_idx {
            let stored: f64 = rec.get(bp_i).unwrap_or("").parse()
                .map_err(|e| format!("load_basis: row {} basis_bp: {}", i, e))?;
            let drift = (stored - fresh).abs();
            if !drift.is_nan() && drift > worst_drift {
                worst_drift = drift;
                worst_drift_row = i;
            }
        }
        records.push(BasisRecord {
            time_s: t, close_spot: s, close_perp: p, basis_bp: fresh,
        });
    }
    if stored_bp_idx.is_some() && worst_drift > recompute_basis_tol_bp {
        return Err(format!(
            "load_basis: basis_bp drifted by {:.4}bp at row {}; possible \
             forward-looking smoothing in source feed",
            worst_drift, worst_drift_row
        ));
    }
    records.sort_by_key(|r| r.time_s);
    Ok(BasisFrame {
        records,
        instrument_pair: instrument_pair.to_string(),
    })
}

pub fn basis_at(frame: &BasisFrame, t_s: i64) -> Option<BasisRecord> {
    let mut last: Option<BasisRecord> = None;
    for rec in &frame.records {
        if rec.time_s <= t_s {
            last = Some(*rec);
        } else {
            break;
        }
    }
    last
}
