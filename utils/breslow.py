import numpy as np
from scipy.interpolate import interp1d

class BreslowEstimator:
    """
    Breslow Estimator for the Cumulative Baseline Hazard Function.
    
    该类实现了基于 Breslow 方法的基准累积风险函数估计。
    适用于 Cox Proportional Hazards 模型和 DeepSurv 等深度生存模型。
    
    数学原理:
    H0(t) = sum_{ti <= t} ( d_i / sum_{j in R_i} exp(f(x_j)) )
    
    Attributes:
    -----------
    times_ : np.array
        训练集中发生事件的唯一时间点。
    cum_baseline_hazard_ : np.array
        对应 times_ 的累积基准风险值 H0(t)。
    baseline_survival_ : sksurv.functions.StepFunction (conceptually)
        用于预测生存函数的插值器。
    """

    def fit(self, log_risk_scores, events, times):
        """
        拟合 Breslow 估计器。

        Parameters
        ----------
        log_risk_scores : array-like, shape = (n_samples,)
            模型的线性预测输出 (Linear Predictor), 即 log(Hazard Ratio)。
            对于 XGBoost，这对应 predict(..., output_margin=True)。
            对于 DeepSurv，这对应网络的直接输出。
            注意：不要传入 exp 后的值！
            
        events : array-like, shape = (n_samples,)
            事件指示器，1 表示发生事件，0 表示删失。
            
        times : array-like, shape = (n_samples,)
            生存时间或删失时间。
        """
        
        log_risk_scores = np.array(log_risk_scores)
        events = np.array(events).astype(int)
        times = np.array(times)

        
        risk_scores = np.exp(log_risk_scores)

        
        
        
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        sorted_events = events[sort_idx]
        sorted_risk_scores = risk_scores[sort_idx]

        
        unique_times = np.unique(sorted_times)
        
        
        
        
        
        
        
        risk_sum_reverse = np.cumsum(sorted_risk_scores[::-1])
        risk_at_risk_full = risk_sum_reverse[::-1]

        
        
        
        time_indices = np.searchsorted(sorted_times, unique_times)
        risk_denominators = risk_at_risk_full[time_indices]

        
        
        
        
        
        
        
        
        intervals = np.append(time_indices, len(sorted_times))
        event_counts = np.array([
            np.sum(sorted_events[intervals[i]:intervals[i+1]]) 
            for i in range(len(unique_times))
        ])

        
        
        non_zero_events = event_counts > 0
        
        valid_times = unique_times[non_zero_events]
        valid_denominators = risk_denominators[non_zero_events]
        valid_counts = event_counts[non_zero_events]

        # dH = d_i / sum(risk)
        dH = valid_counts / valid_denominators
        
        
        H0 = np.cumsum(dH)

        
        
        self.times_ = np.concatenate([[0.0], valid_times])
        self.cum_baseline_hazard_ = np.concatenate([[0.0], H0])

        return self

    def get_survival_function(self, log_risk_scores):
        """
        返回生存函数对象 (Step Function)。
        S(t|x) = exp(-H0(t) * exp(x*beta))
        """
        
        
        pass 

    def predict_survival_probabilities(self, log_risk_scores, time_points):
        """
        预测特定时间点的生存概率。
        
        Parameters
        ----------
        log_risk_scores : array-like
            新样本的线性预测值 (X * beta)。
        time_points : float or array-like
            需要预测的一个或多个时间点。
            
        Returns
        -------
        probs : np.array
            形状为 (n_samples, n_time_points) 的生存概率矩阵。
        """
        log_risk_scores = np.array(log_risk_scores)
        risk_scores = np.exp(log_risk_scores)
        
        time_points = np.atleast_1d(time_points)
        
        
        
        # side='right' return i such that a[i-1] <= v < a[i]
        
        idx = np.searchsorted(self.times_, time_points, side='right') - 1
        
        
        
        
        
        
        
        # H0_at_times shape: (n_time_points,)
        H0_at_times = self.cum_baseline_hazard_[idx]
        
        
        # S(t, x) = exp( -H0(t) * risk_score )
        
        exponential_component = np.outer(risk_scores, H0_at_times)
        surv_probs = np.exp(-exponential_component)
        
        return surv_probs