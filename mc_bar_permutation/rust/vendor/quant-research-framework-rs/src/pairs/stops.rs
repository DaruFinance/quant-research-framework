//! Spread-aware stop-loss families (Rust mirror).

#![cfg(feature = "pairs")]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    ZMultiple,
    HalfLifeMultiple,
    Breakdown,
}

#[derive(Debug, Clone)]
pub struct StopDecision {
    pub fired: bool,
    pub reason: Option<StopReason>,
    pub detail: String,
}

impl StopDecision {
    fn idle() -> Self {
        Self {
            fired: false,
            reason: None,
            detail: String::new(),
        }
    }

    fn fired(reason: StopReason, detail: String) -> Self {
        Self {
            fired: true,
            reason: Some(reason),
            detail,
        }
    }
}

/// Fire if `|z| > z_mult` at bar `t_idx`, with z computed on the
/// trailing `window` bars (excluding the current bar from mean/std).
pub fn z_multiple_stop(
    spread: &[f64],
    t_idx: usize,
    window: usize,
    z_mult: f64,
) -> StopDecision {
    if t_idx < window {
        return StopDecision::idle();
    }
    let seg: Vec<f64> = spread[(t_idx - window)..t_idx]
        .iter()
        .filter(|v| !v.is_nan())
        .copied()
        .collect();
    if seg.len() < 2 {
        return StopDecision::idle();
    }
    let n = seg.len() as f64;
    let mean: f64 = seg.iter().sum::<f64>() / n;
    // numpy std(ddof=1).
    let var: f64 =
        seg.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt() + 1e-12;
    let z = (spread[t_idx] - mean) / std;
    if z.abs() > z_mult {
        StopDecision::fired(
            StopReason::ZMultiple,
            format!("|z|={:.3} > {}", z.abs(), z_mult),
        )
    } else {
        StopDecision::idle()
    }
}

pub fn half_life_multiple_stop(
    entry_idx: usize,
    t_idx: usize,
    half_life: f64,
    hl_mult: f64,
) -> StopDecision {
    if half_life <= 0.0 || !half_life.is_finite() {
        return StopDecision::idle();
    }
    let held = (t_idx - entry_idx) as f64;
    if held >= hl_mult * half_life {
        StopDecision::fired(
            StopReason::HalfLifeMultiple,
            format!(
                "held={} >= {}*hl={:.1}",
                held as i64,
                hl_mult,
                hl_mult * half_life
            ),
        )
    } else {
        StopDecision::idle()
    }
}

pub fn breakdown_trigger_stop(
    beta_prev: f64,
    beta_new: f64,
    beta_jump: f64,
) -> StopDecision {
    if beta_prev == 0.0 {
        return StopDecision::idle();
    }
    let rel = (beta_new - beta_prev).abs() / beta_prev.abs();
    if rel > beta_jump {
        StopDecision::fired(
            StopReason::Breakdown,
            format!("|Δβ|/|β|={:.3} > {}", rel, beta_jump),
        )
    } else {
        StopDecision::idle()
    }
}
