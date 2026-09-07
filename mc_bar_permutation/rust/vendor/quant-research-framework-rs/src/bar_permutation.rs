//! Aronson-style OHLCV bar permutation.
//!
//! Close log returns and complete intrabar templates are shuffled with two
//! independent permutations. Timestamps remain fixed. Volume moves with the
//! template that supplied the open/high/low ratios, preserving their joint
//! empirical distribution. Non-positive OHLC is rejected because logarithmic
//! reconstruction is not defined for instruments such as negative-price WTI.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::Bar;

fn shuffled_indices(length: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut indices = (0..length).collect::<Vec<_>>();
    for index in (1..length).rev() {
        let other = rng.random_range(0..=index);
        indices.swap(index, other);
    }
    indices
}

pub fn validate_bars(bars: &[Bar]) -> Result<(), String> {
    if bars.len() < 2 {
        return Err(format!("bar permutation requires at least 2 bars, got {}", bars.len()));
    }
    for (index, bar) in bars.iter().enumerate() {
        let ohlc = [bar.open, bar.high, bar.low, bar.close];
        if ohlc.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(format!(
                "bar {index} contains non-positive or non-finite OHLC; log-return bar permutation cannot represent negative-price data"
            ));
        }
        if !bar.volume.is_finite() || bar.volume < 0.0 {
            return Err(format!("bar {index} contains invalid volume"));
        }
        if bar.high < bar.open.max(bar.close)
            || bar.low > bar.open.min(bar.close)
            || bar.low > bar.high
        {
            return Err(format!("bar {index} violates OHLC geometry"));
        }
        if index > 0 && bar.time_unix <= bars[index - 1].time_unix {
            return Err(format!("timestamps must be strictly increasing at bar {index}"));
        }
    }
    Ok(())
}

pub fn permute_bars(bars: &[Bar], seed: u64) -> Result<Vec<Bar>, String> {
    validate_bars(bars)?;
    let count = bars.len();
    let close_returns = bars
        .windows(2)
        .map(|pair| (pair[1].close / pair[0].close).ln())
        .collect::<Vec<_>>();
    let templates = bars
        .iter()
        .map(|bar| {
            (
                (bar.open / bar.close).ln(),
                (bar.high / bar.close).ln(),
                (bar.low / bar.close).ln(),
                bar.volume,
            )
        })
        .collect::<Vec<_>>();

    let mut rng = StdRng::seed_from_u64(seed);
    let return_order = shuffled_indices(count - 1, &mut rng);
    let template_order = shuffled_indices(count, &mut rng);
    let mut closes = Vec::with_capacity(count);
    closes.push(bars[0].close);
    let mut accumulated = 0.0f64;
    for source_index in return_order {
        accumulated += close_returns[source_index];
        closes.push(bars[0].close * accumulated.exp());
    }

    let mut result = Vec::with_capacity(count);
    for index in 0..count {
        let (open_ratio, high_ratio, low_ratio, volume) = templates[template_order[index]];
        let close = closes[index];
        result.push(Bar {
            time_unix: bars[index].time_unix,
            open: close * open_ratio.exp(),
            high: close * high_ratio.exp(),
            low: close * low_ratio.exp(),
            close,
            volume,
        });
    }
    validate_bars(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars() -> Vec<Bar> {
        vec![
            Bar { time_unix: 1, open: 10.0, high: 12.0, low: 9.0, close: 11.0, volume: 100.0 },
            Bar { time_unix: 2, open: 11.0, high: 13.0, low: 10.0, close: 12.0, volume: 200.0 },
            Bar { time_unix: 3, open: 12.0, high: 14.0, low: 11.0, close: 13.0, volume: 300.0 },
            Bar { time_unix: 4, open: 13.0, high: 15.0, low: 12.0, close: 14.0, volume: 400.0 },
        ]
    }

    #[test]
    fn permutation_is_seeded_and_preserves_time_and_geometry() {
        let source = bars();
        let first = permute_bars(&source, 17).unwrap();
        let second = permute_bars(&source, 17).unwrap();
        assert_eq!(
            first.iter().map(|bar| bar.close.to_bits()).collect::<Vec<_>>(),
            second.iter().map(|bar| bar.close.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            first.iter().map(|bar| bar.time_unix).collect::<Vec<_>>(),
            source.iter().map(|bar| bar.time_unix).collect::<Vec<_>>()
        );
        let mut source_volumes = source.iter().map(|bar| bar.volume as i64).collect::<Vec<_>>();
        let mut permuted_volumes = first.iter().map(|bar| bar.volume as i64).collect::<Vec<_>>();
        source_volumes.sort_unstable();
        permuted_volumes.sort_unstable();
        assert_eq!(source_volumes, permuted_volumes);
        validate_bars(&first).unwrap();
    }

    #[test]
    fn non_positive_prices_are_rejected() {
        let mut source = bars();
        source[2].low = -1.0;
        let error = match permute_bars(&source, 1) {
            Ok(_) => panic!("negative price was accepted"),
            Err(error) => error,
        };
        assert!(error.contains("negative-price data"));
    }
}
