//! Panel data loader (Rust mirror).
//!
//! Read N per-asset OHLC[+V] CSVs, inner-join on `time`, return a
//! `PanelData` whose `data` field is a `(time, asset, field)`
//! `ndarray::Array3<f64>`. Mirrors the Python `load_panel` contract.

#![cfg(feature = "panel")]

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use ndarray::Array3;

pub const REQUIRED_FIELDS: &[&str] = &["open", "high", "low", "close"];
pub const OPTIONAL_FIELDS: &[&str] = &["volume"];

#[derive(Debug)]
pub enum PanelError {
    /// A source CSV is malformed (missing column, bad time dtype,
    /// duplicate timestamps, empty intersection).
    Schema(String),
    /// The inner-join produced a non-uniform time grid. `ts` is the
    /// first offending timestamp.
    Gap { message: String, ts: i64 },
    /// Underlying IO / CSV-parse failure.
    Io(String),
}

impl std::fmt::Display for PanelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanelError::Schema(s) => write!(f, "panel schema: {}", s),
            PanelError::Gap { message, ts } => {
                write!(f, "panel gap at ts={}: {}", ts, message)
            }
            PanelError::Io(s) => write!(f, "panel io: {}", s),
        }
    }
}

impl std::error::Error for PanelError {}

impl From<std::io::Error> for PanelError {
    fn from(e: std::io::Error) -> Self {
        PanelError::Io(e.to_string())
    }
}
impl From<csv::Error> for PanelError {
    fn from(e: csv::Error) -> Self {
        PanelError::Io(e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct PanelData {
    /// Inner-join timestamp grid, sorted ascending. UNIX seconds.
    pub times: Vec<i64>,
    /// Asset symbols. Order matches input order in `load_panel`.
    pub assets: Vec<String>,
    /// Field names in the data array's axis-2 order. Always starts with
    /// the canonical OHLC; "volume" appended if every asset had it.
    pub fields: Vec<String>,
    /// `(time, asset, field)` array. Shape: (times.len(), assets.len(),
    /// fields.len()).
    pub data: Array3<f64>,
    /// Inferred uniform spacing in seconds (modal delta).
    pub interval_seconds: i64,
}

#[derive(Debug)]
struct AssetTable {
    times: Vec<i64>,
    // Each inner vec is a column in REQUIRED_FIELDS order.
    fields: HashMap<String, Vec<f64>>,
}

fn parse_csv(path: &Path, asset: &str) -> Result<AssetTable, PanelError> {
    let mut rdr = csv::ReaderBuilder::new().from_path(path)?;
    let headers: Vec<String> = rdr
        .headers()?
        .iter()
        .map(|s| s.to_string())
        .collect();
    if !headers.iter().any(|h| h == "time") {
        return Err(PanelError::Schema(format!(
            "asset {:?} ({:?}) missing 'time' column; got {:?}",
            asset, path, headers
        )));
    }
    for required in REQUIRED_FIELDS {
        if !headers.iter().any(|h| h == *required) {
            return Err(PanelError::Schema(format!(
                "asset {:?} ({:?}) missing required column {:?}; got {:?}",
                asset, path, required, headers
            )));
        }
    }
    let mut col_idx: HashMap<String, usize> = HashMap::new();
    for (i, h) in headers.iter().enumerate() {
        col_idx.insert(h.clone(), i);
    }
    let mut times: Vec<i64> = Vec::new();
    let mut fields: HashMap<String, Vec<f64>> = HashMap::new();
    for f in REQUIRED_FIELDS {
        fields.insert(f.to_string(), Vec::new());
    }
    for f in OPTIONAL_FIELDS {
        if headers.iter().any(|h| h == f) {
            fields.insert(f.to_string(), Vec::new());
        }
    }
    for rec in rdr.records() {
        let rec = rec?;
        let t: i64 = rec[col_idx["time"]].parse().map_err(|_| {
            PanelError::Schema(format!(
                "asset {:?} ({:?}): non-integer time value {:?}",
                asset, path, &rec[col_idx["time"]]
            ))
        })?;
        times.push(t);
        for (name, col) in fields.iter_mut() {
            let v: f64 = rec[col_idx[name]].parse().map_err(|_| {
                PanelError::Schema(format!(
                    "asset {:?} ({:?}): non-numeric {} value {:?}",
                    asset, path, name, &rec[col_idx[name]]
                ))
            })?;
            col.push(v);
        }
    }
    // Sort by time, detecting duplicates.
    let n = times.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| times[i]);
    let sorted_times: Vec<i64> = order.iter().map(|&i| times[i]).collect();
    for w in sorted_times.windows(2) {
        if w[0] == w[1] {
            return Err(PanelError::Schema(format!(
                "asset {:?} ({:?}): duplicate timestamp at ts={}",
                asset, path, w[0]
            )));
        }
    }
    let sorted_fields: HashMap<String, Vec<f64>> = fields
        .into_iter()
        .map(|(name, col)| (name, order.iter().map(|&i| col[i]).collect()))
        .collect();
    Ok(AssetTable {
        times: sorted_times,
        fields: sorted_fields,
    })
}

fn infer_interval(times: &[i64]) -> Result<i64, PanelError> {
    if times.len() < 2 {
        return Err(PanelError::Schema(format!(
            "panel needs >= 2 timestamps after inner-join; got {}",
            times.len()
        )));
    }
    let mut counter: HashMap<i64, usize> = HashMap::new();
    for w in times.windows(2) {
        *counter.entry(w[1] - w[0]).or_insert(0) += 1;
    }
    Ok(*counter.iter().max_by_key(|(_, c)| *c).unwrap().0)
}

/// Load N per-asset OHLC CSVs and inner-join them on `time`.
///
/// `paths` is an ordered slice of `(asset_symbol, csv_path)`. Ordering
/// is preserved in `PanelData.assets`. The csv crate handles header
/// parsing; we tolerate float-valued times that round to int.
pub fn load_panel(paths: &[(String, PathBuf)]) -> Result<PanelData, PanelError> {
    if paths.is_empty() {
        return Err(PanelError::Schema("paths is empty".to_string()));
    }
    let mut tables: Vec<AssetTable> = Vec::with_capacity(paths.len());
    for (asset, p) in paths {
        tables.push(parse_csv(p, asset)?);
    }

    // Inner-join: intersection of timestamp sets.
    let mut common: BTreeSet<i64> = tables[0].times.iter().copied().collect();
    for t in &tables[1..] {
        let set: BTreeSet<i64> = t.times.iter().copied().collect();
        common = common.intersection(&set).copied().collect();
    }
    if common.is_empty() {
        return Err(PanelError::Schema(
            "inner-join across assets produced empty timestamp set".to_string(),
        ));
    }
    let times: Vec<i64> = common.into_iter().collect();
    let interval = infer_interval(&times)?;
    for (i, w) in times.windows(2).enumerate() {
        let d = w[1] - w[0];
        if d != interval {
            return Err(PanelError::Gap {
                message: format!(
                    "ts {} -> {} = {}s (expected {}s, modal delta)",
                    w[0], w[1], d, interval
                ),
                ts: w[1],
            });
        }
        let _ = i;
    }

    // Decide final field set: REQUIRED ∪ (volume iff every asset has it).
    let have_volume = tables
        .iter()
        .all(|t| t.fields.contains_key("volume"));
    let mut fields: Vec<String> = REQUIRED_FIELDS.iter().map(|s| s.to_string()).collect();
    if have_volume {
        fields.push("volume".to_string());
    }

    let n_t = times.len();
    let n_a = tables.len();
    let n_f = fields.len();
    let mut data: Array3<f64> = Array3::zeros((n_t, n_a, n_f));

    for (ai, table) in tables.iter().enumerate() {
        // Build map from timestamp -> row index in this asset's table.
        let mut idx_of: HashMap<i64, usize> = HashMap::with_capacity(table.times.len());
        for (i, &t) in table.times.iter().enumerate() {
            idx_of.insert(t, i);
        }
        for (ti, &t) in times.iter().enumerate() {
            let row = idx_of[&t];
            for (fi, fname) in fields.iter().enumerate() {
                data[[ti, ai, fi]] = table.fields[fname][row];
            }
        }
    }

    Ok(PanelData {
        times,
        assets: paths.iter().map(|(a, _)| a.clone()).collect(),
        fields,
        data,
        interval_seconds: interval,
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_paths() -> Vec<(String, PathBuf)> {
        let base = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR not set")
            + "/tests/fixtures/sources";
        vec![
            ("SOL".to_string(), format!("{}/SOLUSDT_1h_30000_31000.csv", base).into()),
            ("BTC".to_string(), format!("{}/BTCUSDT_1h_jan_feb_2024.csv", base).into()),
            ("ETH".to_string(), format!("{}/ETHUSDT_1h_jan_feb_2024.csv", base).into()),
        ]
    }

    #[test]
    fn ds_panel_3_shape() {
        let panel = load_panel(&fixture_paths()).unwrap();
        assert_eq!(panel.assets, vec!["SOL", "BTC", "ETH"]);
        assert_eq!(panel.fields, vec!["open", "high", "low", "close"]);
        assert_eq!(panel.times.len(), 1000);
        assert_eq!(panel.data.shape(), &[1000, 3, 4]);
        assert_eq!(panel.interval_seconds, 3600);
    }

    #[test]
    fn idempotent_loads() {
        let a = load_panel(&fixture_paths()).unwrap();
        let b = load_panel(&fixture_paths()).unwrap();
        assert_eq!(a.times, b.times);
        assert_eq!(a.assets, b.assets);
        assert_eq!(a.fields, b.fields);
        assert!(a.data == b.data);
    }

    #[test]
    fn empty_paths_errors() {
        let r = load_panel(&[]);
        assert!(matches!(r, Err(PanelError::Schema(_))));
    }
}
