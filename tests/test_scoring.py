import unittest
import pandas as pd
import numpy as np
import sys
import os

# 将 src 添加到路径中以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.left_brain.scoring import calculate_tech_score, apply_winsorization

class TestScoring(unittest.TestCase):
    
    def test_apply_winsorization(self):
        # 测试裁剪和映射
        s = pd.Series([4.0, 3.0, 0.0, -3.0, -5.0])
        winsorized = apply_winsorization(s, lower=-3.0, upper=3.0)
        
        # 4.0 应该被裁剪到 3.0，然后映射到 1.0
        self.assertEqual(winsorized.iloc[0], 1.0)
        # 3.0 应该被映射到 1.0
        self.assertEqual(winsorized.iloc[1], 1.0)
        # 0.0 应该被映射到 0.0
        self.assertEqual(winsorized.iloc[2], 0.0)
        # -3.0 应该被映射到 -1.0
        self.assertEqual(winsorized.iloc[3], -1.0)
        # -5.0 应该被裁剪到 -3.0，然后映射到 -1.0
        self.assertEqual(winsorized.iloc[4], -1.0)

    def test_calculate_tech_score_weights(self):
        # 测试权重分配
        # 默认权重: RSI_Z: 0.4, MACD_Z: 0.3, BB_Z: 0.3
        df = pd.DataFrame({
            'RSI_Z': [2.0, 4.0],  # 4.0 应该被截断为 3.0
            'MACD_Z': [0.0, 0.0],
            'BB_Z': [0.0, 0.0]
        })
        
        score = calculate_tech_score(df)
        
        # 第一行: (2.0 * 0.4 + 0.0 * 0.3 + 0.0 * 0.3) / 3 = 0.8 / 3
        self.assertAlmostEqual(score.iloc[0], 0.8 / 3.0)
        
        # 第二行: (3.0 * 0.4 + 0.0 * 0.3 + 0.0 * 0.3) / 3 = 1.2 / 3 = 0.4
        self.assertAlmostEqual(score.iloc[1], 0.4)

    def test_calculate_tech_score_missing_cols(self):
        # 测试列缺失时的权重重新分配
        # 只有 RSI_Z，权重自动变为 1.0
        df = pd.DataFrame({
            'RSI_Z': [1.5, 4.0]
        })
        score = calculate_tech_score(df)
        
        # 第一行: 1.5 * 1.0 / 3 = 0.5
        self.assertAlmostEqual(score.iloc[0], 0.5)
        # 第二行: 3.0 * 1.0 / 3 = 1.0
        self.assertAlmostEqual(score.iloc[1], 1.0)

if __name__ == '__main__':
    unittest.main()
