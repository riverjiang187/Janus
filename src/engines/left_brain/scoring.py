import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_winsorization(series: pd.Series, lower: float = -3.0, upper: float = 3.0) -> pd.Series:
    """
    极值处理 (Winsorization)：
    1. 将数值限制在 [lower, upper] 区间。
    2. 将该区间映射到 [-1.0, 1.0]。
    
    默认：[-3, 3] -> [-1, 1] (除以 3)
    """
    clipped = series.clip(lower=lower, upper=upper)
    # 映射到 [-1, 1]
    # 如果范围是对称的 [-A, A]，映射公式为 x / A
    # 这里我们简化处理，假设 lower = -upper
    return clipped / upper

def calculate_tech_score(df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """
    根据各指标的 Z-Score 计算加权平均得分，并进行缩尾处理。
    修正后的逻辑：
    1. 各因子独立截断到 [-3.0, 3.0]。
    2. 加权平均。
    3. 最终映射并锁定在 [-1.0, 1.0]。
    
    Args:
        df: 包含各 Z-Score 列的 DataFrame
        weights: 权重字典。默认为：RSI_Z (0.4), MACD_Z (0.3), BB_Z (0.3)。
        
    Returns:
        Tech_Score: 锁定在 [-1, 1] 之间的最终评分。
    """
    if weights is None:
        weights = {
            'RSI_Z': 0.4,
            'MACD_Z': 0.3,
            'BB_Z': 0.3
        }
    
    # 检查列是否存在
    available_cols = [col for col in weights.keys() if col in df.columns]
    if not available_cols:
        logging.warning("DataFrame 中未发现任何用于打分的 Z-Score 列。")
        return pd.Series(index=df.index, dtype=float)
    
    # 如果部分列缺失，重新归一化剩余权重
    active_weights = {col: weights[col] for col in available_cols}
    total_weight = sum(active_weights.values())
    normalized_weights = {col: w / total_weight for col, w in active_weights.items()}
    
    # 1. & 2. 分项截断再加权
    # 按照业务要求：必须先对每一个单因子独立进行 [-3.0, 3.0] 的极值截断，然后再进行加权平均。
    weighted_sum = pd.Series(0.0, index=df.index)
    for col, weight in normalized_weights.items():
        # 单因子截断
        clipped_feature = df[col].clip(lower=-3.0, upper=3.0)
        weighted_sum += clipped_feature * weight
        
    # 3. 最终输出映射并锁定
    # 将 [-3, 3] 映射到 [-1, 1]：除以 3.0
    final_score = weighted_sum / 3.0
    # 锁定在 [-1, 1]
    final_score = final_score.clip(lower=-1.0, upper=1.0)
    
    return final_score

if __name__ == "__main__":
    # 简单测试逻辑
    data = {
        'RSI_Z': [1.0, 2.0, 4.0, -5.0, 0.0],
        'MACD_Z': [0.5, -1.0, 2.0, -2.0, 0.0],
        'BB_Z': [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    test_df = pd.DataFrame(data)
    score = calculate_tech_score(test_df) # 使用默认权重
    print("Test Scores:\n", score)
