//! Funding × OI joint-move model (Rust mirror).

#![cfg(feature = "carry")]

use crate::carry::funding::FundingFrame;
use crate::carry::models::base::SignalEmission;
use crate::carry::oi::OIFrame;

#[derive(Debug, Clone)]
pub struct FundingOICointegrationModel {
    pub window: usize,
    pub scale: f64,
}

impl FundingOICointegrationModel {
    pub fn new(window: usize, scale: f64) -> Result<Self, String> {
        if window < 5 {
            return Err("window must be >= 5".to_string());
        }
        Ok(Self { window, scale })
    }

    pub fn signal_at(
        &self,
        funding: &FundingFrame,
        oi: &OIFrame,
        t_s: i64,
    ) -> SignalEmission {
        let model = "funding_oi_coint";
        let f_events: Vec<&crate::carry::funding::FundingEvent> = funding
            .events
            .iter()
            .filter(|e| e.time_s <= t_s)
            .collect();
        if f_events.is_empty() || f_events.len() <= self.window {
            return SignalEmission::flat(t_s, model, 0.0);
        }
        let recent_f: Vec<&crate::carry::funding::FundingEvent> = f_events
            [(f_events.len() - self.window - 1)..]
            .to_vec();

        // For each recent funding time, find most recent OI value at-or-before.
        let mut oi_values: Vec<f64> = Vec::with_capacity(recent_f.len());
        for fev in &recent_f {
            let mut last: Option<f64> = None;
            for r in &oi.records {
                if r.time_s <= fev.time_s {
                    last = Some(r.open_interest);
                } else {
                    break;
                }
            }
            match last {
                Some(v) => oi_values.push(v),
                None => return SignalEmission::flat(t_s, model, 0.0),
            }
        }

        let f_arr: Vec<f64> = recent_f.iter().map(|e| e.rate).collect();
        let f_curr = *f_arr.last().unwrap();
        let o_curr = *oi_values.last().unwrap();
        let f_base = &f_arr[..f_arr.len() - 1];
        let o_base = &oi_values[..oi_values.len() - 1];
        let nf = f_base.len() as f64;
        let no = o_base.len() as f64;
        let f_mu: f64 = f_base.iter().sum::<f64>() / nf;
        let o_mu: f64 = o_base.iter().sum::<f64>() / no;
        let f_var: f64 = f_base.iter().map(|v| (v - f_mu).powi(2)).sum::<f64>() / nf;
        let o_var: f64 = o_base.iter().map(|v| (v - o_mu).powi(2)).sum::<f64>() / no;
        let f_sd = f_var.sqrt();
        let o_sd = o_var.sqrt();
        if f_sd == 0.0 || o_sd == 0.0 {
            return SignalEmission::flat(t_s, model, 0.0);
        }
        let z_f = (f_curr - f_mu) / f_sd;
        let z_o = (o_curr - o_mu) / o_sd;
        let sign = |x: f64| -> f64 {
            if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
        };
        let joint = sign(z_f) * sign(z_o) * z_f.abs().min(z_o.abs()) * self.scale;
        let mut inputs = std::collections::BTreeMap::new();
        inputs.insert("z_f".to_string(), z_f);
        inputs.insert("z_o".to_string(), z_o);
        if joint == 0.0 {
            return SignalEmission {
                time_s: t_s,
                direction: 0,
                strength: 0.0,
                inputs,
                model,
            };
        }
        inputs.insert("joint".to_string(), joint);
        let direction = if joint > 0.0 { -sign(z_f) as i32 } else { 0 };
        SignalEmission {
            time_s: t_s,
            direction,
            strength: joint.abs(),
            inputs,
            model,
        }
    }
}
