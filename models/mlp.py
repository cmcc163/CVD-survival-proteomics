import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.basemodel_torch import BaseModelTorch
from utils.survival_tools import NegativeLogLikelihood, CIndexMetric
from utils.breslow import BreslowEstimator
import random
import os

def seed_everything(seed=42):
    # 1. Python & OS
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 4. PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        
    
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
    Custom implementation for the standard multi-layer perceptron
    Fully customized fit method for Regression, Classification and Survival Analysis.
'''

class MLP(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        seed_everything(args.seed)
        
        if self.args.objective == "Survival":
            self.output_dim = 1
            
            self.criterion = NegativeLogLikelihood()
            self.metric = CIndexMetric
            
            self.breslow = BreslowEstimator()
        elif self.args.objective == "regression":
            self.output_dim = 1
            self.criterion = nn.MSELoss()
        elif self.args.objective == "classification":
            self.output_dim = self.args.num_classes
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.objective == "binary":
            self.output_dim = 1
            self.criterion = nn.BCEWithLogitsLoss()

        
        self.model = MLP_Model(
            n_layers=self.params["n_layers"], 
            input_dim=self.args.num_features,
            hidden_dim=self.params["hidden_dim"], 
            output_dim=self.output_dim,
            dropout=self.params.get("dropout", 0.0), 
            task=self.args.objective
        )

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        
        
        
        
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

        
        X_tensor = torch.tensor(X_np).float().to(self.device)
        y_tensor = torch.tensor(y_np).to(self.device)

        
        if X_val is not None:
            X_val_tensor = torch.tensor(X_val).float().to(self.device)
            y_val_tensor = torch.tensor(y_val).to(self.device)
        else:
            X_val_tensor, y_val_tensor = None, None

        
        if self.args.objective == "classification":
            y_tensor = y_tensor.long()
            if y_val_tensor is not None:
                y_val_tensor = y_val_tensor.long()
        else:
            
            y_tensor = y_tensor.float()
            if y_val_tensor is not None:
                y_val_tensor = y_val_tensor.float()

        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=False)
        else:
            val_loader = None

        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        
        train_loss_history = []
        val_score_history = []  
        best_epoch = 0

        
        if self.args.objective == "Survival":
            
            best_val_score = -float("inf")
            maximize_score = True
            metric_calculator = CIndexMetric() 
            metric_name = "C-Index"
        else:
            
            best_val_score = float("inf")
            maximize_score = False
            metric_name = "Loss"

        
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for i, (batch_X, batch_y) in enumerate(train_loader):
                
                
                # Forward
                out = self.model(batch_X)

                
                if self.args.objective in ["regression", "binary"]:
                    out = out.squeeze()
                
                
                loss = self.criterion(out, batch_y)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            train_loss_history.append(epoch_loss / len(train_loader))

            
            if val_loader is not None:
                self.model.eval()
                current_val_score = 0.0
                
                with torch.no_grad():
                    
                    if self.args.objective == "Survival":
                        val_preds = []
                        val_targets = []
                        for batch_val_X, batch_val_y in val_loader:
                            val_out = self.model(batch_val_X)
                            val_preds.append(val_out)
                            val_targets.append(batch_val_y)
                        
                        if len(val_preds) > 0:
                            
                            val_preds = torch.cat(val_preds)
                            val_targets = torch.cat(val_targets)
                            
                            current_val_score = metric_calculator(val_targets, val_preds)
                        else:
                            current_val_score = 0.0

                    
                    else:
                        val_loss_sum = 0.0
                        val_steps = 0
                        for batch_val_X, batch_val_y in val_loader:
                            val_out = self.model(batch_val_X)
                            
                            if self.args.objective in ["regression", "binary"]:
                                val_out = val_out.squeeze()
                            
                            val_loss_sum += self.criterion(val_out, batch_val_y).item()
                            val_steps += 1
                        
                        current_val_score = val_loss_sum / val_steps if val_steps > 0 else float("inf")

                val_score_history.append(current_val_score)

                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Val {metric_name}: {current_val_score:.5f}")

                
                improved = False
                if maximize_score:
                    
                    if current_val_score > best_val_score:
                        best_val_score = current_val_score
                        improved = True
                else:
                    
                    if current_val_score < best_val_score:
                        best_val_score = current_val_score
                        improved = True

                if improved:
                    best_epoch = epoch
                    self.save_model(filename_extension="best", directory="tmp")
                
                # Check Patience
                if best_epoch + self.args.early_stopping_rounds < epoch:
                    print(f"Validation {metric_name} has not improved for {self.args.early_stopping_rounds} steps!")
                    print("Early stopping applies.")
                    break
        
        
        if val_loader is not None:
            self.load_model(filename_extension="best", directory="tmp")
        
        
        if self.args.objective == "Survival":
            
            self.fit_breslow(X_np, y_np)

        return train_loss_history, val_score_history, best_epoch

    def fit_breslow(self, X_train, y_train):
        """
        基于加载的最佳权重，重新在训练集上推理并拟合 Breslow Estimator。
        """
        print("Fitting Breslow Estimator for MLP...")
        
        
        self.model.eval()

        
        
        log_risk_scores = self.predict_helper(X_train).flatten()
        
        
        # y_train shape: [N, 2] -> [time, event]
        train_times = y_train[:, 0]
        train_events = y_train[:, 1]
        
        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        print(f"Breslow Estimator fitted successfully.")

    def predict(self, X):
        """
        预测函数：
        - Survival: 返回 (risk_scores, event_probs)
        - Classification: 返回 softmax props 或 class (视需求而定，这里返回 probs)
        - Regression: 返回 values
        """
        X = np.array(X, dtype=np.float32)

        if self.args.objective == "Survival":
            
            log_risk_scores = self.predict_helper(X).flatten()
            
            
            time_point = 3652.0 
            
            
            if hasattr(self, 'breslow'):
                surv_probs = self.breslow.predict_survival_probabilities(
                    log_risk_scores, time_point
                ).flatten()
                
                
                event_probs = 1.0 - surv_probs
            else:
                raise AttributeError("Breslow未拟合，请先运行 fit()")
            
            self.predictions = log_risk_scores
            self.prediction_probabilities = event_probs
            return log_risk_scores, event_probs

        else:
            
            return super().predict(X)

    def predict_helper(self, X):
        """
        返回模型的原始输出 (Logits / Linear output)
        """
        self.model.eval()
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.tensor(X).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor)
            if self.args.objective == "binary":
                output = torch.sigmoid(output)
        
        return output.cpu().numpy()

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            # "learning_rate": trial.suggest_float("learning_rate", 5e-4, 2e-2, log = True),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 2e-2, log = True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5)
        }
        return params


class MLP_Model(nn.Module):

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, task, dropout=0.0):
        super().__init__()
        self.task = task
        self.layers = nn.ModuleList()
        
        
        self.dropout = nn.Dropout(p=dropout)

        # Input Layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden Layers
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        # Output Layer (No activation, No dropout usually on output)
        x = self.output_layer(x)

        return x