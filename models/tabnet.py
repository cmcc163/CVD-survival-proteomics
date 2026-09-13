from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
import numpy as np
import torch
from models.basemodel_torch import BaseModelTorch
from utils.io_utils import save_model_to_file, load_model_from_file
from utils.survival_tools import NegativeLogLikelihood,CIndexMetric
import copy
from pytorch_tabnet.utils import filter_weights
from utils.breslow import BreslowEstimator
'''
    TabNet: Attentive Interpretable Tabular Learning (https://arxiv.org/pdf/1908.07442.pdf)

    See the implementation: https://github.com/dreamquark-ai/tabnet
'''


class TabNet(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        
        
        model_args = self.params.copy()

        
        model_args['verbose'] = 0
        model_args["n_a"] = model_args["n_d"]

        model_args["cat_idxs"] = args.cat_idx if args.cat_idx else []
        model_args["cat_dims"] = args.cat_dims

        model_args["device_name"] = self.device
        model_args["scheduler_fn"] = torch.optim.lr_scheduler.StepLR
        model_args["scheduler_params"] = {
            "step_size": 10,
            "gamma": 0.9
        }
        
        
        
        
        lr = model_args.pop("learning_rate", 2e-2)
        
        model_args["optimizer_fn"] = torch.optim.Adam
        model_args["optimizer_params"] = dict(lr=lr, weight_decay=1e-5)
        
        model_args["seed"] = args.seed
        
        if args.objective == "regression":
            self.model = TabNetRegressor(**model_args)
            self.metric = ["rmse"]
        elif args.objective == "classification" or args.objective == "binary":
            self.model = TabNetClassifier(**model_args)
            self.metric = ["auc"]
        elif args.objective == "Survival":
            
            self.model = TabNetSurvival(**model_args)
            
            self.metric = [CIndexMetric]
            self.breslow = BreslowEstimator()

    def fit(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        
        X = X.astype(np.float32)
        if self.args.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)
        elif self.args.objective == "Survival":
            
            y = y.astype(np.float32)
            if y_val is not None:
                y_val = y_val.astype(np.float32)
                
        
        self.model.fit(X, y, 
                    eval_set=[(X_val, y_val)], 
                    eval_name=["eval"], 
                    eval_metric=self.metric,
                    max_epochs=self.args.epochs, 
                    patience=self.args.early_stopping_rounds,
                    batch_size=self.args.batch_size,
                    num_workers=0,
                    weights=0)
        
        
        if self.args.objective == "Survival":
            
            # from utils.breslow import BreslowEstimator
            # if not hasattr(self, 'breslow'): self.breslow = BreslowEstimator()
            
            print("Fitting Breslow Estimator for TabNet...")
            
            
            
            
            log_risk_scores = self.model.predict(X).flatten()
            
            
            
            train_times = y[:, 0]
            train_events = y[:, 1]
            
            
            self.breslow.fit(log_risk_scores, train_events, train_times)
        
        
        history = self.model.history
        best_epoch = self.model.best_epoch
        
        return history['loss'], history["eval_c_index"], best_epoch
    
    def fit_breslow(self, X_train, y_train):
        """
        基于加载的权重，重新在训练集上推理并拟合 Breslow Estimator。
        适配 TabNet 的数据输入逻辑。
        """
        if self.args.objective != "Survival":
            return

        print("Fitting Breslow Estimator for TabNet...")

        
        
        if not hasattr(self, 'breslow'):
            self.breslow = BreslowEstimator()

        
        
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        
        print(f"Predicting risk scores on {len(X_train)} samples...")
        preds = self.model.predict(X_train)
        
        
        log_risk_scores = preds.flatten()

        
        
        train_times = y_train[:, 0]
        train_events = y_train[:, 1]

        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        
        print(f"Breslow Estimator fitted successfully on {len(log_risk_scores)} samples.")
        
    def predict(self, X):
        """
        预测函数：如果是生存分析，返回 (risk_scores, event_probs)
        """
        import numpy as np
        
        
        X = X.astype(np.float32)

        
        if self.args.objective == "Survival":
            
            
            
            log_risk_scores = self.model.predict(X).flatten()
            
            
            
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
            pred = self.model.predict(X)
            
            if self.args.objective == "regression":
                pred = pred.flatten()
            
            self.predictions = pred
            return pred
    
    def fit_best_n_epochs(self, X, y, best_n_epochs):
        """
        使用确定的最佳 Epoch 数进行全量训练。
        不划分验证集，不进行 Early Stopping。
        如果是生存分析，训练结束后必须重新拟合 Breslow Estimator。
        """
        
        if self.args.objective == "regression":
            y = y.reshape(-1, 1)
        
        X = X.astype(np.float32)

        
        
        self.model.fit(
            X, y, 
            eval_set=[],                    
            eval_name=[],                   
            eval_metric=[],                 
            max_epochs=int(best_n_epochs),  
            patience=0,                     
            batch_size=self.args.batch_size,
            num_workers=0,
            weights=0
        )
        
        if self.args.objective == "Survival":
            print("Refitting Breslow Estimator on full data (TabNet)...")
            
            
            
            log_risk_scores = self.model.predict(X).flatten()
            
            
            
            train_times = y[:, 0]
            train_events = y[:, 1]
            
            
            
            if hasattr(self, 'breslow'):
                self.breslow.fit(log_risk_scores, train_events, train_times)
            else:
                
                from utils.breslow import BreslowEstimator
                self.breslow = BreslowEstimator()
                self.breslow.fit(log_risk_scores, train_events, train_times)
        
    def predict_helper(self, X):
        X = np.array(X, dtype=float)

        if self.args.objective == "regression":
            return self.model.predict(X)
        elif self.args.objective == "classification" or self.args.objective == "binary":
            return self.model.predict_proba(X)

    def save_model(self, filename_extension=""):
        save_model_to_file(self.model, self.args, filename_extension)

    def load_model(self, filename_extension=""):
        self.model = load_model_from_file(self.model, self.args, filename_extension)

    def get_model_size(self):
        # To get the size, the model has be trained for at least one epoch
        model_size = sum(t.numel() for t in self.model.network.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_d": trial.suggest_categorical("n_d", [8,16,32]),
            "n_steps": trial.suggest_int("n_steps", 3, 8),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "cat_emb_dim": trial.suggest_int("cat_emb_dim", 2, 3),
            "n_independent": trial.suggest_int("n_independent", 2, 4),
            "n_shared": trial.suggest_int("n_shared", 2, 4),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-3, 1, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["entmax"]),
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 2e-2, log=True)
        }
        return params

    def attribute(self, X: np.ndarray, y: np.ndarray, stategy=""):
        """ Generate feature attributions for the model input.
            Only strategy are supported: default ("") 
            Return attribution in the same shape as X.
        """
        X = np.array(X, dtype=float)
        attributions = self.model.explain(torch.tensor(X, dtype=torch.float32))[0]
        return attributions



class TabNetSurvival(TabNetRegressor):
    def __post_init__(self):
        super().__post_init__()
        
        self._task = 'survival_analysis'
        self._default_loss = NegativeLogLikelihood()
        self._default_metric = 'c_index'
          
    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " +\
                  f"but y_train.shape={y_train.shape} given.\n" +\
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = 1
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def _predict_epoch(self, name, loader):
        """
        重写 _predict_epoch 以确保 y_true 也被移动到 GPU。
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            
            
            
            y = y.to(self.device)
            
            list_y_true.append(y)
            list_y_score.append(scores)

        
        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return
    
    def _predict_batch(self, X):
        """
        重写 _predict_batch 方法，使其返回 GPU 上的 PyTorch 张量。
        原始方法会将其转换为 CPU 上的 NumPy 数组。
        """
        X = X.to(self.device).float()

        
        scores, _ = self.network(X)

        
        return scores.detach()
    
    def stack_batches(self, list_y_true, list_y_score):
        """
        重写 stack_batches 方法，使用 torch.cat 在 GPU 上进行拼接。
        """
        
        y_true = torch.cat(list_y_true, dim=0)
        y_score = torch.cat(list_y_score, dim=0)

        return y_true, y_score