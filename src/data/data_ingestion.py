import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import os
from pathlib import Path

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 自动获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_data_path(symbol: str, interval: str, start: str, end: str) -> Path:
    """
    生成数据文件的存储路径。
    """
    s = start if start else "all"
    e = end if end else "latest"
    filename = f"{symbol}_{interval}_{s}_{e}.csv"
    return DATA_DIR / filename

def fetch_data(symbol: str, start: str = None, end: str = None, interval: str = '1d', retries: int = 3) -> pd.DataFrame:
    """
    获取 OHLCV 数据。
    逻辑：先检查本地缓存，若无则从 yfinance 下载（带指数退避重试），若失败则返回模拟数据。
    """
    # 1. 检查本地缓存
    file_path = get_data_path(symbol, interval, start, end)
    if file_path.exists():
        logging.info(f"发现本地缓存文件: {file_path}，直接读取。")
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if not data.empty:
                # 确保 yfinance 2.x 的列名兼容性（如果缓存是从旧版存的）
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data
        except Exception as e:
            logging.error(f"读取本地文件失败: {e}")

    # 2. 从 yfinance 下载
    logging.info(f"正在从 yfinance 下载 {symbol} 数据，范围：{start} 到 {end}，周期：{interval}")
    for i in range(retries):
        try:
            data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
            if not data.empty:
                # 修复 yfinance 2.x 可能出现的 MultiIndex 列名问题
                if isinstance(data.columns, pd.MultiIndex):
                    # 如果只有单列（如只有一个 symbol），yfinance 也会返回 MultiIndex
                    # 我们取第一层索引，即 OHLCV 名称
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                
                # 标准列名映射，确保即便 yfinance 返回小写或 MultiIndex 也能统一
                mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adj close': 'Adj Close'
                }
                data.columns = [mapping.get(str(c).lower(), c) for c in data.columns]
                
                # 成功后自动保存
                save_data(data, symbol, interval, start, end)
                return data
            else:
                logging.warning(f"下载的 {symbol} 数据为空（尝试 {i+1}/{retries}）。")
        except Exception as e:
            logging.error(f"下载数据时出错 (尝试 {i+1}/{retries}): {e}")
        
        if i < retries - 1:
            # 指数退避 (Exponential Backoff): 2s, 4s, 8s...
            wait_time = 2 ** (i + 1)
            logging.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
            
    # 3. 容错处理：生成模拟数据
    logging.warning("!!! WARNING: ALL DOWNLOAD RETRIES FAILED. USING MOCK DATA !!!")
    return generate_mock_data(symbol, start, end)

def save_data(df: pd.DataFrame, symbol: str, interval: str, start: str, end: str):
    """
    将清洗后的数据保存到本地 /data/ 文件夹。
    """
    if df.empty:
        logging.warning("数据为空，跳过保存。")
        return

    # 确保文件夹存在
    if not DATA_DIR.exists():
        logging.info(f"创建数据目录: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = get_data_path(symbol, interval, start, end)
    df.to_csv(file_path)
    logging.info(f"数据已保存至: {file_path}")

def generate_mock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    生成模拟的 OHLCV 数据，标记为 is_mock=True。
    """
    dates = pd.date_range(start=start if start else "2023-01-01", 
                          end=end if end else "2023-12-31", 
                          freq='D')
    n = len(dates)
    np.random.seed(42)
    # 模拟随机游走
    price = 100 + np.cumsum(np.random.randn(n))
    data = pd.DataFrame({
        'Open': price,
        'High': price + 1,
        'Low': price - 1,
        'Close': price,
        'Volume': np.random.randint(1000, 10000, size=n),
        'is_mock': True # 显式标记为模拟数据
    }, index=dates)
    data.index.name = 'Date'
    return data

def calculate_log_returns(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
    """
    计算 Log_Return = np.log(Close / Close.shift(1))。
    保留首行 NaN，由 handle_missing_and_inf 统一处理。
    """
    if df.empty or column not in df.columns:
        logging.warning(f"DataFrame 为空或未找到列 {column}。")
        return df
    
    df = df.copy()
    # 确保没有零或负数以防 log 报错
    values = df[column]
    if (values <= 0).any():
        logging.warning(f"列 {column} 中包含零或负值，计算 Log_Return 可能会产生 NaN/Inf。")
    
    df['Log_Return'] = np.log(df[column] / df[column].shift(1))
    return df

def calculate_ma_distance(df: pd.DataFrame, column: str = 'Close', window: int = 20) -> pd.DataFrame:
    """
    计算 Distance_from_MA = (Close - MA) / MA。
    """
    df = df.copy()
    ma = df[column].rolling(window=window).mean()
    df[f'Distance_from_MA_{window}'] = (df[column] - ma) / ma
    return df

def handle_missing_and_inf(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理缺失值 (NaN) 和无穷大 (Inf)。
    1. 将 Inf 替换为 NaN。
    2. 删除包含 NaN 的行（或者前向填充，视策略而定）。
    根据 Phase 1 要求，我们主要进行清洗。
    """
    df = df.copy()
    # 替换 Inf 为 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 记录 NaN 数量
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logging.info(f"发现 {nan_count} 个 NaN，进行删除处理。")
        df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # 测试代码
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    df = fetch_data(symbol, start=start_date, end=end_date)
    if not df.empty:
        # 保存原始下载的数据
        save_data(df, symbol, '1d', start_date, end_date)
        
        df = calculate_log_returns(df)
        df = calculate_ma_distance(df, window=20)
        df = handle_missing_and_inf(df)
        
        print("\n--- Data Sample ---")
        print(df.head())
        print(df.tail())
        
        if 'is_mock' in df.columns:
            logging.warning("!!! 注意：当前使用的是模拟数据 !!!")
        else:
            logging.info("使用的是真实行情数据。")
            
        logging.info("1.1 数据获取、清洗与工程化改进验证成功。")
