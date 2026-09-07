//! Persistent-funding-sign carry model (Rust mirror).

#![cfg(feature = "carry")]

use crate::carry::funding::FundingFrame;
use crate::carry::models::base::SignalEmission;

#[derive(Debug, Clone)]
pub struct PersistentFundingSignModel {
    pub min_streak: usize,
}

impl PersistentFundingSignModel {
    pub fn new(min_streak: usize) -> Result<Self, String> {
        if min_streak < 1 {
            return Err("min_streak must be >= 1".to_string());
        }
        Ok(Self { min_streak })
    }

    pub fn signal_at(&self, frame: &FundingFrame, t_s: i64) -> SignalEmission {
        let model = "persistent_sign";
        let mut slc: Vec<f64> = Vec::new();
        for ev in &frame.events {
            if ev.time_s <= t_s {
                slc.push(ev.rate);
            } else {
                break;
            }
        }
        if slc.is_empty() {
            return SignalEmission::flat(t_s, model, 0.0);
        }
        let last_rate = *slc.last().unwrap();
        let last_sign = if last_rate > 0.0 { 1 } else if last_rate < 0.0 { -1 } else { 0 };
        if last_sign == 0 {
            return SignalEmission::flat(t_s, model, 0.0);
        }
        let mut streak = 1usize;
        // Walk backwards from second-to-last.
        if slc.len() >= 2 {
            for v in slc.iter().rev().skip(1) {
                let s = if *v > 0.0 { 1 } else if *v < 0.0 { -1 } else { 0 };
                if s == last_sign {
                    streak += 1;
                } else {
                    break;
                }
            }
        }
        let mut inputs = std::collections::BTreeMap::new();
        inputs.insert("streak".to_string(), streak as f64);
        inputs.insert("last_rate".to_string(), last_rate);
        if streak < self.min_streak {
            return SignalEmission {
                time_s: t_s,
                direction: 0,
                strength: streak as f64 / self.min_streak as f64,
                inputs,
                model,
            };
        }
        SignalEmission {
            time_s: t_s,
            direction: -last_sign,
            strength: streak as f64,
            inputs,
            model,
        }
    }
}
