//! Monte Carlo modes for completed trade returns.
//!
//! Permutation shuffles without replacement. Resampling bootstraps with
//! replacement. Each invocation selects exactly one null model.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const DEFAULT_TRADE_RUNS: usize = 1_000;
pub const DEFAULT_SEED: u64 = 42;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MonteCarloMode {
    Permutation,
    Resampling,
}

impl MonteCarloMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Permutation => "permutation",
            Self::Resampling => "resampling",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "permutation" => Ok(Self::Permutation),
            "resampling" => Ok(Self::Resampling),
            _ => Err(format!(
                "unknown Monte Carlo mode {value:?}; choose permutation or resampling"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MetricRank {
    pub name: &'static str,
    pub actual: f64,
    pub percentile_midrank: f64,
    pub less: usize,
    pub ties: usize,
}

#[derive(Clone, Debug)]
pub struct MonteCarloResult {
    pub mode: MonteCarloMode,
    pub seed: u64,
    pub requested_runs: usize,
    pub completed_runs: usize,
    pub sharpe_convention: &'static str,
    pub drawdown_convention: &'static str,
    pub metrics: Vec<MetricRank>,
    pub equity_finals: Vec<f64>,
    pub loss_percent: f64,
    pub drawdown_over_80_percent: f64,
}

#[derive(Clone, Copy)]
struct Values {
    roi: f64,
    pf: f64,
    win_rate: f64,
    expectancy: f64,
    sharpe: f64,
    max_drawdown: f64,
    consistency: f64,
    equity_final: f64,
}

impl Values {
    fn get(self, index: usize) -> f64 {
        match index {
            0 => self.roi,
            1 => self.pf,
            2 => self.win_rate,
            3 => self.expectancy,
            4 => self.sharpe,
            5 => self.max_drawdown,
            6 => self.consistency,
            _ => unreachable!(),
        }
    }
}

const METRIC_NAMES: [&str; 7] = [
    "ROI",
    "PF",
    "WinRate",
    "Exp",
    "Sharpe",
    "MaxDrawdown",
    "Consistency",
];

fn metric_values(sample: &[f64], use_forex: bool) -> Values {
    let n = sample.len();
    let roi: f64 = sample.iter().sum();
    let wins_sum: f64 = sample.iter().filter(|&&v| v > 0.0).sum();
    let losses_sum: f64 = sample.iter().filter(|&&v| v <= 0.0).map(|v| -v).sum();
    let wins = sample.iter().filter(|&&v| v > 0.0).count();
    let losses = n - wins;
    let win_rate = wins as f64 / n as f64;
    let mean_win = if wins > 0 { wins_sum / wins as f64 } else { 0.0 };
    let mean_loss = if losses > 0 { losses_sum / losses as f64 } else { 0.0 };
    let expectancy = mean_win * win_rate - mean_loss * (1.0 - win_rate);
    let mean = roi / n as f64;
    let variance = sample.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let standard_deviation = variance.sqrt();
    let sharpe = if n > 1 && standard_deviation > 0.0 {
        mean / standard_deviation * (n as f64).sqrt()
    } else {
        0.0
    };

    let (equity_final, max_drawdown) = if use_forex {
        let mut equity = 0.0f64;
        let mut high_water = 0.0f64;
        let mut maximum = 0.0f64;
        for value in sample {
            equity += value;
            high_water = high_water.max(equity);
            maximum = maximum.max(high_water - equity);
        }
        (equity, maximum)
    } else {
        let mut equity = 1.0f64;
        let mut high_water = 1.0f64;
        let mut maximum = 0.0f64;
        for value in sample {
            equity += value;
            high_water = high_water.max(equity);
            let drawdown = if high_water > 0.0 {
                (high_water - equity) / high_water
            } else {
                0.0
            };
            maximum = maximum.max(drawdown);
        }
        (equity, maximum)
    };

    let weights = [0.0117, 0.0317, 0.0861, 0.2341, 0.6364];
    let mut segment_sums = [0.0f64; 5];
    let quotient = n / 5;
    let remainder = n % 5;
    let mut start = 0usize;
    for segment in 0..5 {
        let length = quotient + usize::from(segment < remainder);
        segment_sums[segment] = sample[start..start + length].iter().sum();
        start += length;
    }
    let weighted = weights
        .iter()
        .zip(segment_sums.iter())
        .map(|(weight, sum)| weight * sum)
        .sum::<f64>();

    Values {
        roi,
        pf: if losses_sum > 0.0 {
            wins_sum / losses_sum
        } else {
            f64::INFINITY
        },
        win_rate,
        expectancy,
        sharpe,
        max_drawdown,
        consistency: 0.6 * weighted + 0.4 * roi,
        equity_final,
    }
}

fn shuffle(values: &mut [f64], rng: &mut StdRng) {
    for index in (1..values.len()).rev() {
        let other = rng.random_range(0..=index);
        values.swap(index, other);
    }
}

fn tied(value: f64, actual: f64) -> bool {
    if value.is_infinite() || actual.is_infinite() {
        return value == actual;
    }
    let tolerance = 1e-15f64.max(1e-12 * value.abs().max(actual.abs()));
    (value - actual).abs() <= tolerance
}

pub fn run_trade_monte_carlo(
    returns: &[f64],
    mode: MonteCarloMode,
    runs: usize,
    seed: u64,
    use_forex: bool,
) -> Result<MonteCarloResult, String> {
    if returns.is_empty() {
        return Err("returns must not be empty".into());
    }
    if returns.iter().any(|value| !value.is_finite()) {
        return Err("returns must contain only finite values".into());
    }
    if runs == 0 {
        return Err("runs must be positive".into());
    }

    let actual = metric_values(returns, use_forex);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut distributions = (0..METRIC_NAMES.len())
        .map(|_| Vec::with_capacity(runs))
        .collect::<Vec<_>>();
    let mut equity_finals = Vec::with_capacity(runs);

    for _ in 0..runs {
        let sample = match mode {
            MonteCarloMode::Permutation => {
                let mut values = returns.to_vec();
                shuffle(&mut values, &mut rng);
                values
            }
            MonteCarloMode::Resampling => (0..returns.len())
                .map(|_| returns[rng.random_range(0..returns.len())])
                .collect(),
        };
        let values = metric_values(&sample, use_forex);
        for (index, distribution) in distributions.iter_mut().enumerate() {
            distribution.push(values.get(index));
        }
        equity_finals.push(values.equity_final);
    }

    let metrics = METRIC_NAMES
        .iter()
        .enumerate()
        .map(|(index, &name)| {
            let actual_value = actual.get(index);
            let ties = distributions[index]
                .iter()
                .filter(|&&value| tied(value, actual_value))
                .count();
            let less = distributions[index]
                .iter()
                .filter(|&&value| value < actual_value && !tied(value, actual_value))
                .count();
            MetricRank {
                name,
                actual: actual_value,
                percentile_midrank: (less as f64 + 0.5 * ties as f64) / runs as f64 * 100.0,
                less,
                ties,
            }
        })
        .collect();
    let loss_percent = distributions[0].iter().filter(|&&value| value < 0.0).count() as f64
        / runs as f64
        * 100.0;
    let drawdown_over_80_percent = distributions[5]
        .iter()
        .filter(|&&value| value > 0.80)
        .count() as f64
        / runs as f64
        * 100.0;

    Ok(MonteCarloResult {
        mode,
        seed,
        requested_runs: runs,
        completed_runs: runs,
        sharpe_convention: "completed-trade mean/std * sqrt(trade count)",
        drawdown_convention: if use_forex {
            "absolute R from equity 0"
        } else {
            "fractional from equity 1"
        },
        metrics,
        equity_finals,
        loss_percent,
        drawdown_over_80_percent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_preserves_order_invariant_metrics() {
        let returns = [0.02, -0.01, 0.03, -0.015, 0.01];
        let result = run_trade_monte_carlo(
            &returns,
            MonteCarloMode::Permutation,
            25,
            7,
            false,
        )
        .unwrap();
        for name in ["ROI", "PF", "WinRate", "Exp", "Sharpe"] {
            let metric = result.metrics.iter().find(|metric| metric.name == name).unwrap();
            assert_eq!(metric.ties, 25, "{name}");
            assert_eq!(metric.percentile_midrank, 50.0, "{name}");
        }
    }

    #[test]
    fn resampling_is_deterministic_and_changes_order_invariant_metrics() {
        let returns = [0.02, -0.01, 0.03, -0.015, 0.01];
        let first = run_trade_monte_carlo(
            &returns,
            MonteCarloMode::Resampling,
            25,
            9,
            false,
        )
        .unwrap();
        let second = run_trade_monte_carlo(
            &returns,
            MonteCarloMode::Resampling,
            25,
            9,
            false,
        )
        .unwrap();
        assert_eq!(first.equity_finals, second.equity_finals);
        let roi = first.metrics.iter().find(|metric| metric.name == "ROI").unwrap();
        assert!(roi.ties < 25);
    }

    #[test]
    fn no_loss_profit_factor_is_infinite_for_actual_and_samples() {
        let result = run_trade_monte_carlo(
            &[0.01, 0.02, 0.03],
            MonteCarloMode::Permutation,
            10,
            1,
            false,
        )
        .unwrap();
        let profit_factor = result.metrics.iter().find(|metric| metric.name == "PF").unwrap();
        assert!(profit_factor.actual.is_infinite());
        assert_eq!(profit_factor.ties, 10);
    }

    #[test]
    fn forex_uses_absolute_drawdown() {
        let returns = [1.0, -2.0, 0.5];
        let result = run_trade_monte_carlo(
            &returns,
            MonteCarloMode::Permutation,
            1,
            1,
            true,
        )
        .unwrap();
        let drawdown = result
            .metrics
            .iter()
            .find(|metric| metric.name == "MaxDrawdown")
            .unwrap();
        assert_eq!(drawdown.actual, 2.0);
        assert_eq!(result.drawdown_convention, "absolute R from equity 0");
    }
}
