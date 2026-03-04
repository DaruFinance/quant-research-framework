import pandas as pd

def compute_sma(data, length, column='close'):
    length = int(length)   
    return data[column].rolling(window=length).mean()


def compute_ema(data, length):
    return data['close'].ewm(span=length, adjust=False).mean()

def compute_macd(data, fast_length=12, slow_length=26, signal_length=9):
    fast = compute_ema(data, fast_length)
    slow = compute_ema(data, slow_length)
    macd = fast - slow
    sig  = macd.ewm(span=signal_length, adjust=False).mean()
    return macd, sig

def compute_rsi(data, length):
    delta = data['close'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=length-1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length-1, min_periods=length).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(data, length):
    hl = data['high'] - data['low']
    hc = (data['high'] - data['close'].shift()).abs()
    lc = (data['low']  - data['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length).mean()

def compute_stoch(data, length):
    lo = data['low'].rolling(window=length).min()
    hi = data['high'].rolling(window=length).max()
    return 100 * (data['close'] - lo) / (hi - lo)
