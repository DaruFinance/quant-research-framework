use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use quant_research_framework_rs::{
    default_ema_signal, run_frozen_backtest, Bar as EngineBar, Config, Metrics, Trade,
};

const MAGIC: &[u8; 8] = b"QRFMCB01";

#[derive(Clone)]
struct Bar {
    time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

fn validate(bars: &[Bar]) -> Result<(), String> {
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
        if index > 0 && bar.time <= bars[index - 1].time {
            return Err(format!("timestamps must be strictly increasing at bar {index}"));
        }
    }
    Ok(())
}

fn load_csv(path: &str) -> Result<Vec<Bar>, String> {
    let file = File::open(path).map_err(|error| format!("cannot open {path}: {error}"))?;
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .ok_or_else(|| "input CSV is empty".to_string())?
        .map_err(|error| error.to_string())?;
    let columns = header.split(',').map(|value| value.trim()).collect::<Vec<_>>();
    let required = ["time", "open", "high", "low", "close"];
    for name in required {
        if !columns.contains(&name) {
            return Err(format!("input CSV is missing {name:?}"));
        }
    }
    let find = |name: &str| columns.iter().position(|column| *column == name).unwrap();
    let time = find("time");
    let open = find("open");
    let high = find("high");
    let low = find("low");
    let close = find("close");
    let volume = columns.iter().position(|column| *column == "volume");
    let mut bars = Vec::new();
    for (offset, line) in lines.enumerate() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split(',').map(|value| value.trim()).collect::<Vec<_>>();
        let parse_float = |index: usize, name: &str| -> Result<f64, String> {
            fields
                .get(index)
                .ok_or_else(|| format!("row {} is missing {name}", offset + 2))?
                .parse::<f64>()
                .map_err(|error| format!("row {} has invalid {name}: {error}", offset + 2))
        };
        let time_value = fields
            .get(time)
            .ok_or_else(|| format!("row {} is missing time", offset + 2))?
            .parse::<i64>()
            .map_err(|error| format!("row {} has invalid time: {error}", offset + 2))?;
        bars.push(Bar {
            time: time_value,
            open: parse_float(open, "open")?,
            high: parse_float(high, "high")?,
            low: parse_float(low, "low")?,
            close: parse_float(close, "close")?,
            volume: match volume {
                Some(index) => parse_float(index, "volume")?,
                None => 0.0,
            },
        });
    }
    validate(&bars)?;
    Ok(bars)
}

fn shuffled_indices(length: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut indices = (0..length).collect::<Vec<_>>();
    for index in (1..length).rev() {
        let other = rng.random_range(0..=index);
        indices.swap(index, other);
    }
    indices
}

fn permute(bars: &[Bar], seed: u64) -> Result<Vec<Bar>, String> {
    validate(bars)?;
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
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let (open_ratio, high_ratio, low_ratio, volume) = templates[template_order[index]];
        let close = closes[index];
        output.push(Bar {
            time: bars[index].time,
            open: close * open_ratio.exp(),
            high: close * high_ratio.exp(),
            low: close * low_ratio.exp(),
            close,
            volume,
        });
    }
    validate(&output)?;
    Ok(output)
}

fn write_binary(path: &str, bars: &[Bar]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("cannot create {path}: {error}"))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(MAGIC).map_err(|error| error.to_string())?;
    writer
        .write_all(&(bars.len() as u64).to_le_bytes())
        .map_err(|error| error.to_string())?;
    for bar in bars {
        writer.write_all(&bar.time.to_le_bytes()).map_err(|error| error.to_string())?;
        for value in [bar.open, bar.high, bar.low, bar.close, bar.volume] {
            writer.write_all(&value.to_le_bytes()).map_err(|error| error.to_string())?;
        }
    }
    writer.flush().map_err(|error| error.to_string())
}

fn argument(args: &[String], name: &str) -> Result<String, String> {
    let index = args.iter().position(|value| value == name)
        .ok_or_else(|| format!("missing {name}"))?;
    args.get(index + 1).cloned().ok_or_else(|| format!("missing value after {name}"))
}

fn parse_bool(value: &str, name: &str) -> Result<bool, String> {
    match value {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(format!("{name} must be true or false")),
    }
}

fn json_float(value: f64) -> String {
    if value == f64::INFINITY { "\"Infinity\"".into() }
    else if value == f64::NEG_INFINITY { "\"-Infinity\"".into() }
    else if value.is_nan() { "\"NaN\"".into() }
    else { format!("{value:.17}") }
}

fn write_ledger(path: &Path, trades: &[Trade]) -> Result<(), String> {
    let mut writer = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
    writer.write_all(b"QRFMCL01").map_err(|error| error.to_string())?;
    writer.write_all(&(trades.len() as u64).to_le_bytes()).map_err(|error| error.to_string())?;
    for trade in trades {
        writer.write_all(&trade.side.to_le_bytes()).map_err(|error| error.to_string())?;
        writer.write_all(&trade.entry_idx.to_le_bytes()).map_err(|error| error.to_string())?;
        writer.write_all(&trade.exit_idx.to_le_bytes()).map_err(|error| error.to_string())?;
        for value in [trade.entry_price, trade.exit_price, trade.qty, trade.net_pnl,
                      trade.fee, trade.slippage, trade.funding, trade.gross_pnl] {
            writer.write_all(&value.to_le_bytes()).map_err(|error| error.to_string())?;
        }
    }
    writer.flush().map_err(|error| error.to_string())
}

fn write_metrics(path: &Path, seed: u64, bars: usize, metrics: &Metrics) -> Result<(), String> {
    let text = format!(
        "{{\n  \"seed\": {seed},\n  \"bars\": {bars},\n  \"trades\": {trades},\n  \"metrics\": {{\n    \"ROI\": {roi},\n    \"PF\": {pf},\n    \"WinRate\": {win},\n    \"Exp\": {exp},\n    \"Sharpe\": {sharpe},\n    \"MaxDrawdown\": {dd},\n    \"Consistency\": {cons}\n  }}\n}}\n",
        trades = metrics.trades, roi = json_float(metrics.roi), pf = json_float(metrics.pf),
        win = json_float(metrics.win_rate), exp = json_float(metrics.exp),
        sharpe = json_float(metrics.sharpe), dd = json_float(metrics.max_drawdown),
        cons = json_float(metrics.consistency),
    );
    std::fs::write(path, text).map_err(|error| error.to_string())
}

fn run_backtest_mode(args: &[String]) -> Result<(), String> {
    let input = argument(args, "--input")?;
    let output = argument(args, "--output")?;
    let seed = argument(args, "--seed")?.parse::<u64>().map_err(|error| error.to_string())?;
    let lookback = argument(args, "--lookback")?.parse::<usize>().map_err(|error| error.to_string())?;
    if argument(args, "--strategy")? != "ema-crossover" {
        return Err("the built-in Rust worker supports ema-crossover; use the explicit Python fallback for a Python callable".into());
    }
    let source = load_csv(&input)?;
    let permuted = permute(&source, seed)?;
    let bars = permuted.iter().map(|bar| EngineBar {
        time_unix: bar.time, open: bar.open, high: bar.high, low: bar.low,
        close: bar.close, volume: bar.volume,
    }).collect::<Vec<_>>();
    let mut cfg = Config::new();
    cfg.fee_pct = argument(args, "--fee-pct")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?;
    cfg.slippage_pct = argument(args, "--slippage-pct")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?;
    cfg.funding_fee = argument(args, "--funding-fee")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?;
    cfg.account_size = argument(args, "--account-size")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?;
    cfg.position_size = argument(args, "--position-size")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?;
    cfg.use_sl = parse_bool(&argument(args, "--use-sl")?, "--use-sl")?;
    cfg.sl_override = Some(argument(args, "--sl-percentage")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?);
    cfg.use_tp = parse_bool(&argument(args, "--use-tp")?, "--use-tp")?;
    cfg.tp_percentage = argument(args, "--tp-percentage")?.parse().map_err(|error: std::num::ParseFloatError| error.to_string())?;
    cfg.use_forex = parse_bool(&argument(args, "--forex")?, "--forex")?;
    cfg.max_hold_bars = argument(args, "--max-hold-bars")?.parse().map_err(|error: std::num::ParseIntError| error.to_string())?;
    cfg.sharpe_bar = argument(args, "--sharpe-mode")? == "bar";
    let result = run_frozen_backtest(&bars, &cfg, lookback, default_ema_signal)?;
    let output = Path::new(&output);
    std::fs::create_dir_all(output).map_err(|error| error.to_string())?;
    write_ledger(&output.join("ledger.bin"), &result.trades)?;
    write_metrics(&output.join("metrics.json"), seed, bars.len(), &result.metrics)
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("backtest") {
        return run_backtest_mode(&args);
    }
    if args.len() != 4 {
        return Err("usage: qrf-bar-permute INPUT.csv OUTPUT.bin SEED".into());
    }
    let seed = args[3]
        .parse::<u64>()
        .map_err(|error| format!("invalid seed: {error}"))?;
    let bars = load_csv(&args[1])?;
    let output = permute(&bars, seed)?;
    write_binary(&args[2], &output)?;
    println!("bars={} seed={} output={}", output.len(), seed, args[2]);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
