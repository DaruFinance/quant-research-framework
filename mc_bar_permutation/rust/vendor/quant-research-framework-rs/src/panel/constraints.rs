//! Portfolio-level constraints (Rust mirror).
//!
//! Single-asset weight cap (iterative redistribution to strictly
//! under-cap legs, avoiding the cap-oscillation loop) + gross
//! leverage cap (uniform rescaling). Both idempotent. Pure pointwise.
//!
//! Mirrors ``backtester.panel.constraints`` in Python.

#![cfg(feature = "panel")]

/// Apply optional caps to a weight vector. Either or both can be
/// `None` (no-op). Returns a new vector; input not mutated.
pub fn apply_constraints(
    weights: &[f64],
    single_asset_max: Option<f64>,
    gross_lev_max: Option<f64>,
) -> Result<Vec<f64>, String> {
    let mut w: Vec<f64> = weights.to_vec();

    if let Some(cap) = single_asset_max {
        if !(cap > 0.0 && cap <= 1.0) {
            return Err(format!(
                "single_asset_max must lie in (0, 1]; got {}",
                cap
            ));
        }
        w = cap_single_asset(&w, cap);
    }

    if let Some(gross_max) = gross_lev_max {
        if gross_max <= 0.0 {
            return Err(format!("gross_lev_max must be > 0; got {}", gross_max));
        }
        let gross: f64 = w.iter().map(|x| x.abs()).sum();
        if gross > gross_max {
            let scale = gross_max / gross;
            for x in w.iter_mut() {
                *x *= scale;
            }
        }
    }

    Ok(w)
}

fn cap_single_asset(w: &[f64], cap: f64) -> Vec<f64> {
    const TOL: f64 = 1e-12;
    const MAX_ITER: usize = 100;
    if w.is_empty() {
        return Vec::new();
    }
    let signs: Vec<f64> = w.iter().map(|x| x.signum()).collect();
    let mut abs_w: Vec<f64> = w.iter().map(|x| x.abs()).collect();

    for _ in 0..MAX_ITER {
        let mut excess = 0.0;
        let mut over_count = 0usize;
        for x in abs_w.iter_mut() {
            if *x > cap + TOL {
                excess += *x - cap;
                *x = cap;
                over_count += 1;
            }
        }
        if over_count == 0 {
            break;
        }
        // Strictly-under legs only, avoids oscillation when legs at
        // cap absorb residual and re-cross.
        let mut under_sum = 0.0;
        let mut under_idx: Vec<usize> = Vec::new();
        for (i, x) in abs_w.iter().enumerate() {
            if *x < cap - TOL {
                under_sum += *x;
                under_idx.push(i);
            }
        }
        if under_idx.is_empty() {
            // No headroom; drop the residual.
            break;
        }
        if under_sum == 0.0 {
            let per = excess / under_idx.len() as f64;
            for i in &under_idx {
                abs_w[*i] = per;
            }
        } else {
            for i in &under_idx {
                abs_w[*i] += excess * (abs_w[*i] / under_sum);
            }
        }
    }
    // Defensive final clip for tiny floating overshoot.
    for x in abs_w.iter_mut() {
        if *x > cap {
            *x = cap;
        }
    }
    signs.iter().zip(abs_w.iter()).map(|(s, a)| s * a).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn no_cap_returns_input_unchanged() {
        let w = vec![0.4, -0.3, 0.3];
        let out = apply_constraints(&w, None, None).unwrap();
        assert_eq!(out, w);
    }

    #[test]
    fn single_asset_cap_redistributes() {
        let w = vec![0.6, 0.2, 0.2];
        let out = apply_constraints(&w, Some(0.5), None).unwrap();
        assert!(approx(&out, &[0.5, 0.25, 0.25], 1e-10));
    }

    #[test]
    fn single_asset_cap_drops_excess_when_no_headroom() {
        let w = vec![0.5, 0.3, 0.2];
        let out = apply_constraints(&w, Some(0.3), None).unwrap();
        assert!(approx(&out, &[0.3, 0.3, 0.3], 1e-10));
        let gross: f64 = out.iter().map(|x| x.abs()).sum();
        assert!((gross - 0.9).abs() < 1e-10);
    }

    #[test]
    fn single_asset_cap_idempotent() {
        let w = vec![0.5, 0.3, 0.2];
        let once = apply_constraints(&w, Some(0.3), None).unwrap();
        let twice = apply_constraints(&once, Some(0.3), None).unwrap();
        assert!(approx(&once, &twice, 1e-12));
    }

    #[test]
    fn gross_leverage_cap_scales_down() {
        let w = vec![2.0, -1.0, 1.5];
        let out = apply_constraints(&w, None, Some(3.0)).unwrap();
        let gross: f64 = out.iter().map(|x| x.abs()).sum();
        assert!((gross - 3.0).abs() < 1e-10);
    }

    #[test]
    fn gross_leverage_cap_idempotent() {
        let w = vec![2.0, -1.0, 1.5];
        let once = apply_constraints(&w, None, Some(3.0)).unwrap();
        let twice = apply_constraints(&once, None, Some(3.0)).unwrap();
        assert!(approx(&once, &twice, 1e-12));
    }

    #[test]
    fn rejects_invalid_caps() {
        assert!(apply_constraints(&[0.5, 0.5], Some(0.0), None).is_err());
        assert!(apply_constraints(&[0.5, 0.5], Some(1.5), None).is_err());
        assert!(apply_constraints(&[1.0, 1.0], None, Some(0.0)).is_err());
    }

    #[test]
    fn preserves_signs() {
        let w = vec![0.5, -0.4, 0.1];
        let out = apply_constraints(&w, Some(0.3), None).unwrap();
        for (a, b) in out.iter().zip(w.iter()) {
            assert_eq!(a.signum(), b.signum());
        }
    }
}
