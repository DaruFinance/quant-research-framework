use super::*;

fn trade(side: i8, entry_idx: i32, exit_idx: i32, pnl: f64) -> Trade {
    Trade {
        side,
        entry_idx,
        exit_idx,
        entry_price: 0.0,
        exit_price: 0.0,
        qty: 0.0,
        pnl,
        leg_id: 0,
        trade_group_id: 0,
        fee: 0.0,
        slippage: 0.0,
        funding: 0.0,
        gross_pnl: pnl,
        net_pnl: pnl,
    }
}

#[test]
fn export_trades_flushes_and_appends_without_repeating_header() {
    let path = std::env::temp_dir().join(format!(
        "qrf-export-{}-{:?}.csv",
        std::process::id(),
        std::thread::current().id()
    ));
    let path_str = path.to_str().unwrap();
    let bars = vec![
        Bar::ohlc(1_700_000_000, 10.0, 11.0, 9.0, 10.5),
        Bar::ohlc(1_700_000_060, 10.5, 12.0, 10.0, 11.0),
        Bar::ohlc(1_700_000_120, 11.0, 11.5, 9.5, 10.0),
    ];

    export_trades(
        &[trade(1, 0, 1, 1.25)],
        &bars,
        "ema",
        "W01",
        "IS",
        path_str,
        true,
    );
    let first = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        first,
        "strategy,window,sample,side,entry_time,open_entry,high_entry,low_entry,close_entry,exit_time,open_exit,high_exit,low_exit,close_exit,pnl\n\
         ema,W01,IS,long,1700000000,10,11,9,10.5,1700000060,10.5,12,10,11,1.25\n"
    );

    export_trades(
        &[trade(-1, 1, 2, -2.5)],
        &bars,
        "ema",
        "W01",
        "OOS",
        path_str,
        false,
    );
    let appended = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        appended,
        format!(
            "{}{}",
            first, "ema,W01,OOS,short,1700000060,10.5,12,10,11,1700000120,11,11.5,9.5,10,-2.5\n"
        )
    );
    assert_eq!(appended.matches("strategy,window,sample").count(), 1);
    std::fs::remove_file(path).unwrap();
}
