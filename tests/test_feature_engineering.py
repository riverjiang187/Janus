import unittest
import pandas as pd
import numpy as np
import sys
import os

# 将 src 添加到路径中以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineering import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    robust_z_score,
    add_technical_features
)

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        # 创建 100 天的模拟数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        close = 100 + np.cumsum(np.random.randn(100))
        self.df = pd.DataFrame({
            'Open': close,
            'High': close + 1,
            'Low': close - 1,
            'Close': close,
            'Volume': np.random.randint(100, 1000, 100)
        }, index=dates)

    def test_calculate_rsi(self):
        rsi = calculate_rsi(self.df)
        self.assertEqual(len(rsi), 100)
        # RSI 应该在 0-100 之间
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())

    def test_calculate_macd(self):
        macd, signal = calculate_macd(self.df)
        self.assertEqual(len(macd), 100)
        self.assertEqual(len(signal), 100)

    def test_calculate_bollinger_bands(self):
        bb_p = calculate_bollinger_bands(self.df)
        self.assertEqual(len(bb_p), 100)
        # 正常的 BB %B 大多在 0-1 之间，但也可能超出

    def test_calculate_atr(self):
        atr = calculate_atr(self.df)
        self.assertEqual(len(atr), 100)
        self.assertTrue((atr.dropna() > 0).all())

    def test_robust_z_score(self):
        # 创建一个带有极值的序列
        data = pd.Series([10, 11, 12, 13, 14, 100]) # 100 是异常值
        z = robust_z_score(data)
        
        # 验证中位数附近的 Z-Score 是否较小
        self.assertLess(abs(z.iloc[2]), 2.0)
        # 验证异常值的 Z-Score
        self.assertGreater(abs(z.iloc[5]), 3.0)

    def test_add_technical_features(self):
        df_feat = add_technical_features(self.df)
        expected_cols = ['RSI', 'MACD', 'MACD_Signal', 'BB_Percent', 'ATR', 'RSI_Z', 'MACD_Z', 'BB_Z']
        for col in expected_cols:
            self.assertIn(col, df_feat.columns)

if __name__ == '__main__':
    unittest.main()
