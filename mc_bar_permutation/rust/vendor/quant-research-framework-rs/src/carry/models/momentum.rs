//! Funding-momentum carry model (Rust mirror).

#![cfg(feature = "carry")]

use crate::carry::funding::FundingFrame;
use crate::carry::models::base::SignalEmission;

#[derive(Debug, Clone)]
pub struct FundingMomentumModel {
    pub window: usize,
    pub z_thresh: f64,
}

impl FundingMomentumModel {
    pub fn new(window: usize, z_thresh: f64) -> Result<Self, String> {
        if window < 5 {
            return Err("window must be >= 5".to_string());
        }
        Ok(Self { window, z_thresh })
    }

    pub fn signal_at(&self, frame: &FundingFrame, t_s: i64) -> SignalEmission {
        let model = "funding_momentum";
        let rates: Vec<f64> = frame
            .events
            .iter()
            .filter(|e| e.time_s <= t_s)
            .map(|e| e.rate)
            .collect();
        if rates.is_empty() || rates.len() <= self.window {
            return SignalEmission::flat(t_s, model, 0.0);
        }
        let win = &rates[(rates.len() - self.window - 1)..(rates.len() - 1)];
        let n = win.len() as f64;
        let mu: f64 = win.iter().sum::<f64>() / n;
        // numpy std() defaults to ddof=0.
        let var: f64 = win.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n;
        let sd = var.sqrt();
        if sd == 0.0 || !sd.is_finite() {
            return SignalEmission::flat(t_s, model, 0.0);
        }
        let last = *rates.last().unwrap();
        let z = (last - mu) / sd;
        let mut inputs = std::collections::BTreeMap::new();
        inputs.insert("z".to_string(), z);
        inputs.insert("mu".to_string(), mu);
        inputs.insert("sd".to_string(), sd);
        if z.abs() < self.z_thresh {
            return SignalEmission {
                time_s: t_s,
                direction: 0,
                strength: z.abs(),
                inputs,
                model,
            };
        }
        inputs.insert("rate".to_string(), last);
        let direction = if z > 0.0 { -1 } else if z < 0.0 { 1 } else { 0 };
        SignalEmission {
            time_s: t_s,
            direction,
            strength: z.abs(),
            inputs,
            model,
        }
    }
}
