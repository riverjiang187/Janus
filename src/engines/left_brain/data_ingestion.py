import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import random
from pathlib import Path

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 自动获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def get_data_path(symbol: str, interval: str, start: str, end: str) -> Path:
    """
    生成数据文件的存储路径。
    """
    s = start if start else "all"
    e = end if end else "latest"
    filename = f"{symbol}_{interval}_{s}_{e}.csv"
    return DATA_DIR / filename

def fetch_data(symbol: str, start: str = None, end: str = None, interval: str = '1d', retries: int = 5, proxy: str = None) -> pd.DataFrame:
    """
    获取 OHLCV 数据。
    逻辑：先检查本地缓存，若无则从 yfinance 下载（带指数退避重试），若失败则返回模拟数据。
    
    Args:
        symbol: 股票代码
        start: 开始日期 (YYYY-MM-DD)
        end: 结束日期 (YYYY-MM-DD)
        interval: 时间间隔 (1d, 1wk, 1mo 等)
        retries: 最大重试次数
        proxy: HTTP/HTTPS 代理 (e.g. "http://127.0.0.1:7890")
    """
    # 1. 检查本地缓存
    file_path = get_data_path(symbol, interval, start, end)
    if file_path.exists():
        logging.info(f"发现本地缓存文件: {file_path}，直接读取。")
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if not data.empty:
                # 确保 yfinance 2.x 的列名兼容性
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data
        except Exception as e:
            logging.error(f"读取本地文件失败: {e}")

    # 2. 从 yfinance 下载
    logging.info(f"正在从 yfinance 下载 {symbol} 数据，范围：{start} 到 {end}，周期：{interval}")
    
    for i in range(retries):
        try:
            # yfinance 2.x+ 建议不要手动设置 session，它内部会自动处理并规避检测
            # 如果依然下载失败，通常是由于被 IP 限制。
            # 这里我们通过随机等待来增加一点隐蔽性
            if i > 0:
                jitter_wait = random.uniform(1, 5)
                logging.info(f"下载尝试 {i+1}，先随机等待 {jitter_wait:.2f} 秒...")
                time.sleep(jitter_wait)

            # 尝试主要下载方式
            # 注意：yf.download 2.x 建议通过随机等待和增加重试来提高成功率。
            download_kwargs = {
                "start": start,
                "end": end,
                "interval": interval,
                "progress": False
            }
            # 只有当 proxy 不为 None 时才加入参数
            if proxy:
                download_kwargs["proxy"] = proxy

            data = yf.download(symbol, **download_kwargs)
            
            # 如果主要下载方式失败，尝试备选下载方式 (yf.Ticker.history)
            if data.empty:
                logging.warning(f"yf.download 返回为空，尝试使用 Ticker.history 备选方案...")
                ticker = yf.Ticker(symbol)
                history_kwargs = {
                    "start": start,
                    "end": end,
                    "interval": interval
                }
                if proxy:
                    history_kwargs["proxy"] = proxy
                data = ticker.history(**history_kwargs)

            if not data.empty:
                # 修复 yfinance 2.x 可能出现的 MultiIndex 列名问题
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                
                # 标准列名映射
                mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adj close': 'Adj Close',
                    'dividends': 'Dividends',
                    'stock splits': 'Stock Splits'
                }
                data.columns = [mapping.get(str(c).lower(), c) for c in data.columns]
                
                # 成功后自动保存
                save_data(data, symbol, interval, start, end)
                return data
            else:
                logging.warning(f"下载的 {symbol} 数据为空（尝试 {i+1}/{retries}）。")
        except Exception as e:
            logging.error(f"下载数据时出错 (尝试 {i+1}/{retries}): {e}")
            if "Rate Limit" in str(e) or "Unauthorized" in str(e):
                logging.warning("提示：Yahoo 可能已对您的 IP 进行限流或封禁，建议更换网络或设置代理。")
        
        if i < retries - 1:
            # 指数退避 + 随机抖动 (Jitter)
            wait_time = (2 ** (i + 1)) + random.uniform(1, 3)
            logging.info(f"等待 {wait_time:.2f} 秒后重试...")
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
        logging.warning(f"列 {column} 中包含零或负值，计算 Log_Return 前将其替换为 NaN。")
        # 将非正值替换为 NaN，防止 log(-inf) 或 log(NaN)
        values = values.where(values > 0, np.nan)
    
    df['Log_Return'] = np.log(values / values.shift(1))
    return df

def calculate_ma_distance(df: pd.DataFrame, column: str = 'Close', window: int = 20) -> pd.DataFrame:
    """
    计算 Distance_from_MA = (Close - MA) / MA。
    """
    df = df.copy()
    ma = df[column].rolling(window=window).mean()
    # 使用 .replace(0, np.nan) 防止均线 ma 为 0 时触发除以零错误
    df[f'Distance_from_MA_{window}'] = (df[column] - ma) / ma.replace(0, np.nan)
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
        # fetch_data 内部会自动调用 save_data，此处无需重复调用
        
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
