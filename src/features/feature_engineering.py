import pandas as pd
import numpy as np
import logging
from scipy.stats import median_abs_deviation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算相对强弱指数 (RSI)。
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    计算指数平滑异同移动平均线 (MACD)。
    """
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span=signal, adjust=False).mean()
    return macd, exp3

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2):
    """
    计算布林带 %B 指标。
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    """
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    percent_b = (df['Close'] - lower_band) / (upper_band - lower_band)
    return percent_b

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算平均真实波幅 (ATR)。
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def robust_z_score(series: pd.Series) -> pd.Series:
    """
    实现 Robust Z-Score (中位数去极值)。
    公式：Z = (X - Median(X)) / (MAD(X) * 1.4826)
    """
    median = series.median()
    mad = median_abs_deviation(series, scale='normal', nan_policy='omit')
    # scale='normal' 相当于乘以 1.4826
    # 防止除以零
    if mad == 0:
        return (series - median) * 0
    return (series - median) / mad

from src.data.data_ingestion import calculate_log_returns, calculate_ma_distance

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 DataFrame 添加基础技术面指标。
    """
    df = df.copy()
    
    # 1.1 阶段的特征
    df = calculate_log_returns(df)
    df = calculate_ma_distance(df, window=20)
    
    # 1.2 阶段的基础指标
    df['RSI'] = calculate_rsi(df)
    macd, signal = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['BB_Percent'] = calculate_bollinger_bands(df)
    df['ATR'] = calculate_atr(df)
    
    # 计算 Robust Z-Score 特征
    # 我们对所有用于打分的特征进行 Z-Score 稳健化
    df['RSI_Z'] = robust_z_score(df['RSI'])
    df['MACD_Z'] = robust_z_score(df['MACD'])
    df['BB_Z'] = robust_z_score(df['BB_Percent'])
    df['Log_Return_Z'] = robust_z_score(df['Log_Return'])
    df['MA_Dist_Z'] = robust_z_score(df['Distance_from_MA_20'])
    
    return df

def check_trend_consistency(daily_df: pd.DataFrame) -> pd.Series:
    """
    [多周期校验] 检查日线与周线的趋势一致性。
    1. 计算日线趋势 (e.g., Close > MA20)。
    2. 计算周线趋势 (e.g., Weekly Close > Weekly MA20)。
    3. 将周线趋势广播回日线频率。
    """
    # 确保索引是 DatetimeIndex
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)

    # 1. 日线趋势
    daily_ma = daily_df['Close'].rolling(window=20).mean()
    daily_trend = daily_df['Close'] > daily_ma

    # 2. 周线趋势
    # 重采样为周线 (W-FRI 或 W)
    weekly_df = daily_df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    weekly_ma = weekly_df['Close'].rolling(window=20).mean()
    weekly_trend = weekly_df['Close'] > weekly_ma

    # 3. 对齐：将周线趋势广播回日线
    # reindex 使用 ffill 以便日线能拿到所属周的趋势状态
    aligned_weekly_trend = weekly_trend.reindex(daily_df.index, method='ffill')
    
    # 趋势一致性：日线和周线方向相同
    consistency = (daily_trend == aligned_weekly_trend)
    return consistency

if __name__ == "__main__":
    from src.data.data_ingestion import fetch_data, handle_missing_and_inf
    
    symbol = "AAPL"
    df = fetch_data(symbol, start="2022-01-01", end="2023-12-31")
    
    if not df.empty:
        # 1.1 清洗
        df = handle_missing_and_inf(df)
        
        # 1.2 添加特征
        df = add_technical_features(df)
        df['Trend_Consistency'] = check_trend_consistency(df)
        
        print("\n--- Final Output ---")
        print(df[['Close', 'RSI_Z', 'MACD_Z', 'BB_Z', 'Trend_Consistency']].tail())
        logging.info("1.2 稳健特征工程 (Robust Feature Engineering) 验证成功。")
