import numpy as np
from scipy.optimize import minimize
from typing import List, Dict

class MVOSolver:
    def __init__(self, risk_free_rate: float = 0.02):
        self.rf = risk_free_rate

    def optimize_portfolio(self, expected_returns: List[float], cov_matrix: List[List[float]]) -> Dict[str, float]:
        """执行马科维茨均值-方差优化 (MVO)"""
        returns_array = np.array(expected_returns)
        cov_array = np.array(cov_matrix)
        num_assets = len(returns_array)
        init_guess = np.ones(num_assets) / num_assets
        
        # 目标函数：最小化组合方差
        def portfolio_variance(weights):
            return weights.T @ cov_array @ weights

        # 约束：权重和为1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # 边界：做多约束 (0, 1)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise ValueError(f"MVO Optimization failed: {result.message}")
            
        return {
            "optimal_weights": result.x.tolist(),
            "minimized_variance": float(result.fun)
        }