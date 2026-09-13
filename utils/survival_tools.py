import torch
import torch.nn as nn
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, y):
        """
        y: [Batch, 2] -> y[:, 0]=Time, y[:, 1]=Event
        risk_pred: [Batch, 1] -> 模型输出的风险值
        """
        
        time = y[:, 0].view(-1, 1)
        event = y[:, 1].view(-1, 1)
        risk_pred = risk_pred.view(-1, 1)
        
        
        
        risk_set_mask = (time.T >= time).float()
        
        
        exp_risk_pred = torch.exp(risk_pred)
        
        
        denominator = torch.matmul(risk_set_mask, exp_risk_pred)
        log_denominator = torch.log(denominator + 1e-12)
        
        
        loss_terms = risk_pred - log_denominator
        
        
        loss_for_events = loss_terms * event
        
        
        num_events = torch.sum(event)
        if num_events == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_pred.device)
            
        
        neg_log_loss = -torch.sum(loss_for_events) / num_events
        return neg_log_loss


class CIndexMetric(Metric):
    def __init__(self):
        self._name = "c_index"
        self._maximize = True 

    def __call__(self, y_true, y_score):
        """
        pytorch_tabnet 在验证时传入的是 numpy array。
        这里我们将其转为 tensor 并复用你的逻辑。
        """
        
        
        if not isinstance(y_true, torch.Tensor):
            y_true_t = torch.tensor(y_true)
        else:
            y_true_t = y_true

        
        if not isinstance(y_score, torch.Tensor):
            y_score_t = torch.tensor(y_score)
        else:
            y_score_t = y_score
            
        
        if y_true_t.device != y_score_t.device:
            y_true_t = y_true_t.to(y_score_t.device)

        
        event_times = y_true_t[:, 0]
        event_observed = y_true_t[:, 1]
        
        
        
        predicted_scores = -y_score_t.flatten()
        
        
        c_index = self.concordance_index_torch(event_times, predicted_scores, event_observed)
        
        return c_index.item()

    def concordance_index_torch(self, event_times, predicted_scores, event_observed):
        event_times = event_times.view(-1)
        predicted_scores = predicted_scores.view(-1)
        event_observed = event_observed.view(-1)

        t_i = event_times.unsqueeze(1)
        t_j = event_times.unsqueeze(0)
        
        s_i = predicted_scores.unsqueeze(1)
        s_j = predicted_scores.unsqueeze(0)
        
        e_i = event_observed.unsqueeze(1)
        e_j = event_observed.unsqueeze(0)

        
        pair_mask = ((t_i < t_j) & (e_i == 1)) | ((t_i == t_j) & (e_i == 1) & (e_j == 0))
        
        num_pairs = pair_mask.sum()

        if num_pairs == 0:
            return torch.tensor(0.5, device=event_times.device)
        
        
        num_concordant = (pair_mask & (s_i < s_j)).sum()
        num_tied = (pair_mask & (s_i == s_j)).sum()
        
        c_index = (num_concordant + 0.5 * num_tied) / num_pairs
        
        return c_index

