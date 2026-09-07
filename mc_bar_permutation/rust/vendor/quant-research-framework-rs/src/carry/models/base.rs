//! Common signal-emission contract shared by all carry models.

#![cfg(feature = "carry")]

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct SignalEmission {
    pub time_s: i64,
    pub direction: i32,        // +1 long carry / -1 short / 0 flat
    pub strength: f64,
    pub inputs: BTreeMap<String, f64>,
    pub model: &'static str,
}

impl SignalEmission {
    pub fn flat(time_s: i64, model: &'static str, strength: f64) -> Self {
        Self {
            time_s,
            direction: 0,
            strength,
            inputs: BTreeMap::new(),
            model,
        }
    }
}
