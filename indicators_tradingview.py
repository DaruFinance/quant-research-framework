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

def f_zscore(src: pd.Series, length: int) -> pd.Series:
    """
    Z-score: (src - mean) / std, zero where std is zero.
    """
    mean = src.rolling(window=length, min_periods=length).mean()
    std  = src.rolling(window=length, min_periods=length).std(ddof=0)
    return ((src - mean).where(std != 0, 0)) / std

def f_roc(src: pd.Series, length: int) -> pd.Series:
    """
    Rate of Change: (src - src.shift(length)) / src.shift(length) * 100, NaN if denominator is zero.
    """
    prev = src.shift(length)
    roc  = (src - prev) / prev * 100
    return roc.where(prev != 0)

def f_bias(src: pd.Series, length: int, smooth: int) -> pd.Series:
    """
    EMA of slope: slope = (src - src.shift(length)) / length, then EMA(slope, span=smooth).
    """
    slope = (src - src.shift(length)) / length
    return slope.ewm(span=smooth, adjust=False).mean()

def f_volZ(src: pd.Series, length: int) -> pd.Series:
    """
    Volatility Z-score: compute rolling std of src, 
    then z-score that volatility over the same window.
    """
    vol     = src.rolling(window=length, min_periods=length).std(ddof=0)
    meanVol = vol.rolling(window=length, min_periods=length).mean()
    devVol  = vol.rolling(window=length, min_periods=length).std(ddof=0)
    return ((vol - meanVol).where(devVol != 0, 0)) / devVol

def f_accel(src: pd.Series, length: int) -> pd.Series:
    """
    Second derivative approximation:
      (src - 2*src.shift(length) + src.shift(2*length)) / (length^2).
    """
    shifted1 = src.shift(length)
    shifted2 = src.shift(2 * length)
    return (src - 2*shifted1 + shifted2) / (length * length)

def f_distFromMedian(src: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(0, index=src.index)
    highest = src.rolling(window=length, min_periods=length).max()
    lowest  = src.rolling(window=length, min_periods=length).min()
    median  = (highest + lowest) / 2
    return src - median

