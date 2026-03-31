import unittest
import pandas as pd
import numpy as np
import sys
import os

# 将 src 添加到路径中以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_ingestion import (
    fetch_data, 
    calculate_log_returns, 
    calculate_ma_distance, 
    handle_missing_and_inf,
    generate_mock_data
)

class TestDataIngestion(unittest.TestCase):
    
    def setUp(self):
        self.symbol = "AAPL"
        self.start = "2023-01-01"
        self.end = "2023-01-31"
        self.mock_df = generate_mock_data(self.symbol, self.start, self.end)

    def test_generate_mock_data(self):
        df = generate_mock_data(self.symbol, self.start, self.end)
        self.assertIn('is_mock', df.columns)
        self.assertTrue(df['is_mock'].all())
        self.assertEqual(df.index.name, 'Date')

    def test_calculate_log_returns_nan_preservation(self):
        df = generate_mock_data(self.symbol, self.start, self.end)
        df_ret = calculate_log_returns(df)
        self.assertIn('Log_Return', df_ret.columns)
        self.assertTrue(pd.isna(df_ret['Log_Return'].iloc[0])) # 第一行应该是 NaN
        self.assertFalse(pd.isna(df_ret['Log_Return'].iloc[1]))

    def test_handle_missing_and_inf(self):
        df = self.mock_df.copy()
        df.iloc[0, 0] = np.nan
        df.iloc[1, 1] = np.inf
        
        cleaned_df = handle_missing_and_inf(df)
        self.assertFalse(cleaned_df.isna().any().any())
        self.assertFalse(np.isinf(cleaned_df).any().any())

    def test_fetch_data_mock_awareness(self):
        # 故意触发 yfinance 失败（使用不存在的 symbol 且 retries=0 或通过重试失败）
        # 这里我们的逻辑是重试失败后返回 mock
        # 我们用一个极短的 retries 来加速测试
        df = fetch_data("INVALID_SYMBOL_12345", start=self.start, end=self.end, retries=1)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('is_mock', df.columns)
        self.assertTrue(df['is_mock'].iloc[0])

if __name__ == '__main__':
    unittest.main()
