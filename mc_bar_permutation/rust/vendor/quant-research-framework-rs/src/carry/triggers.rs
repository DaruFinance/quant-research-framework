//! Funding-flip / basis-blowout triggers (Rust mirror).

#![cfg(feature = "carry")]

use crate::carry::basis::BasisFrame;
use crate::carry::funding::FundingFrame;

#[derive(Debug, Clone)]
pub struct TriggerEvent {
    pub time_s: i64,
    pub kind: &'static str,
    pub direction: i32,
    pub prev: f64,
    pub curr: f64,
    pub z: Option<f64>,
    pub sigma: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct FundingFlipTrigger {
    pub min_magnitude: f64,
}

impl FundingFlipTrigger {
    pub fn new(min_magnitude: f64) -> Self {
        Self { min_magnitude }
    }

    pub fn run(&self, frame: &FundingFrame) -> Vec<TriggerEvent> {
        let mut events = Vec::new();
        if frame.events.len() < 2 {
            return events;
        }
        let rates = frame.rates();
        let times = frame.times();
        let sign = |x: f64| -> i32 {
            if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 }
        };
        let mut prev_sign = sign(rates[0]);
        for i in 1..frame.events.len() {
            let curr = rates[i];
            let curr_sign = sign(curr);
            if curr_sign != prev_sign
                && curr_sign != 0
                && curr.abs() >= self.min_magnitude
            {
                events.push(TriggerEvent {
                    time_s: times[i],
                    kind: "funding_flip",
                    direction: curr_sign,
                    prev: rates[i - 1],
                    curr,
                    z: None,
                    sigma: None,
                });
                prev_sign = curr_sign;
            } else if curr_sign != 0 {
                prev_sign = curr_sign;
            }
        }
        events
    }
}

#[derive(Debug, Clone)]
pub struct BasisBlowoutTrigger {
    pub window: usize,
    pub z_thresh: f64,
}

impl BasisBlowoutTrigger {
    pub fn new(window: usize, z_thresh: f64) -> Result<Self, String> {
        if window < 5 {
            return Err("BasisBlowoutTrigger: window must be >= 5".to_string());
        }
        Ok(Self { window, z_thresh })
    }

    pub fn run(&self, frame: &BasisFrame) -> Vec<TriggerEvent> {
        let mut events = Vec::new();
        if frame.records.len() <= self.window {
            return events;
        }
        let basis: Vec<f64> = frame.records.iter().map(|r| r.basis_bp).collect();
        let times: Vec<i64> = frame.records.iter().map(|r| r.time_s).collect();
        for i in self.window..frame.records.len() {
            let window = &basis[(i - self.window)..i];
            let n = window.len() as f64;
            let mu: f64 = window.iter().sum::<f64>() / n;
            let var: f64 =
                window.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt();
            if sd == 0.0 || !sd.is_finite() {
                continue;
            }
            let z = (basis[i] - mu) / sd;
            if z.abs() >= self.z_thresh {
                let direction = if z > 0.0 { 1 } else if z < 0.0 { -1 } else { 0 };
                events.push(TriggerEvent {
                    time_s: times[i],
                    kind: "basis_blowout",
                    direction,
                    prev: mu,
                    curr: basis[i],
                    z: Some(z),
                    sigma: Some(sd),
                });
            }
        }
        events
    }
}
