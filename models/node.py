import time
import shutil
import os
from models import node_lib
from models.basemodel_torch import BaseModelTorch
from models.node_lib.utils import check_numpy, process_in_chunks
from utils.breslow import BreslowEstimator
from utils.survival_tools import NegativeLogLikelihood,CIndexMetric
import torch
import torch.nn as nn
import torch.nn.functional as F
from qhoptim.pyt import QHAdam

import numpy as np

from utils.io_utils import get_output_path

'''
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (https://arxiv.org/abs/1909.06312)

    Code adapted from: https://github.com/Qwicen/node
'''


class NODE(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        layer_dim = int(self.params["total_tree_count"] / self.params["num_layers"])

        if args.objective == "regression":
            self.model = nn.Sequential(
                node_lib.DenseBlock(args.num_features,
                                    # layer_dim=128, num_layers=8, depth=6, tree_dim=3
                                    layer_dim=layer_dim, num_layers=self.params["num_layers"],
                                    depth=self.params["tree_depth"], tree_dim=self.params["tree_output_dim"],
                                    flatten_output=False,
                                    choice_function=node_lib.entmax15, bin_function=node_lib.entmoid15),
                node_lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
            ).to(self.device)

        elif args.objective == "classification" or args.objective == "binary":
            self.model = nn.Sequential(
                node_lib.DenseBlock(args.num_features,
                                    # layer_dim=1024, num_layers=2, depth=6,
                                    layer_dim=layer_dim, num_layers=self.params["num_layers"],
                                    depth=self.params["tree_depth"], tree_dim=args.num_classes + 1,
                                    flatten_output=False,input_dropout=0.1,
                                    choice_function=node_lib.entmax15, bin_function=node_lib.entmoid15),
                node_lib.Lambda(lambda x: x[..., :args.num_classes].mean(dim=-2)),
            ).to(self.device)

        
        elif args.objective == "Survival":
            self.model = nn.Sequential(
                node_lib.DenseBlock(args.num_features,
                                    layer_dim=layer_dim, num_layers=self.params["num_layers"],
                                    depth=self.params["tree_depth"], 
                                    tree_dim=self.params["tree_output_dim"],
                                    flatten_output=False,
                                    choice_function=node_lib.entmax15, bin_function=node_lib.entmoid15),
                node_lib.Lambda(lambda x: x[..., 0].mean(dim=-1)), 
            ).to(self.device)
            
        print("On:", self.device)

        self.trainer = None
        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        
        if self.args.objective == "Survival":
            print("Detected Survival Objective. Redirecting to fit_survival...")
            return self.fit_survival(X, y, X_val, y_val, cross_val_index)
        
        data = node_lib.Dataset(self.args.dataset, random_state=self.args.seed,
                                X_train=np.array(X, dtype=np.float32), y_train=np.array(y, dtype=np.float32),
                                X_valid=np.array(X_val, dtype=np.float32), y_valid=np.array(y_val, dtype=np.float32))

        with torch.no_grad():
            res = self.model(torch.as_tensor(data.X_train[:1000], device=self.device))

        
        experiment_name = f"{self.args.dataset}_{cross_val_index}"

        # -----------------------------------------------------------
        
        
        log_dir = os.path.join('logs', experiment_name)
        if os.path.exists(log_dir):
            print(f"Cleaning up previous experiment logs at: {log_dir}")
            try:
                shutil.rmtree(log_dir) 
            except Exception as e:
                print(f"Warning: Failed to delete {log_dir}. Error: {e}")
        # -----------------------------------------------------------

        if self.args.objective == "regression":
            loss_func = F.mse_loss
        elif self.args.objective == "classification":
            loss_func = F.cross_entropy
            data.y_train = data.y_train.astype(int)
        elif self.args.objective == "binary":
            loss_func = F.binary_cross_entropy_with_logits
            data.y_train = data.y_train.reshape(-1, 1)

        self.trainer = node_lib.Trainer(
            model=self.model, 
            loss_function=loss_func,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(lr=1e-3, nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=False,
            n_last_checkpoints=5
        )
        best_auc = 0.0
        best_step = 0

        loss_history = []
        val_loss_history = []

        early_stopping = self.args.early_stopping_rounds * self.args.logging_period
        # early_stopping = 0
        for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, batch_size=self.args.batch_size, shuffle=True,
                                                  epochs=self.args.epochs, seed=self.args.seed):

            metrics = self.trainer.train_on_batch(*batch, device=self.device)
            loss_history.append(metrics['loss'].item())

            if self.trainer.step % self.args.logging_period == 0:
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')

                if self.args.objective == "regression":
                    metric_val = self.trainer.evaluate_mse(data.X_valid, data.y_valid, device=self.device,
                                                     batch_size=self.args.batch_size)
                    print("Val MSE: %0.5f" % metric_val)
                elif self.args.objective == "classification":
                    metric_val = self.trainer.evaluate_logloss(data.X_valid, data.y_valid, device=self.device,
                                                         batch_size=self.args.batch_size)
                    print("Val LogLoss: %0.5f" % metric_val)
                elif self.args.objective == "binary":
                    metric_val = self.trainer.evaluate_auc(data.X_valid, data.y_valid, device=self.device,
                                                               batch_size=self.args.batch_size)
                
                print("Step: %d, Train Loss: %.5f, Val AUC: %0.5f" % (self.trainer.step, metrics['loss'], metric_val))
                
                val_loss_history.append(metric_val)

                if metric_val > best_auc:
                    best_auc = metric_val
                    best_step = self.trainer.step
                    self.trainer.save_checkpoint(tag='best')

                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()

                if self.trainer.step > best_step + early_stopping:
                    print('BREAK. There is no improvment for {} steps'.format(early_stopping))
                    print("Best step: ", best_step)
                    print("Best Val AUC: %0.5f" % best_auc)
                    break

        self.trainer.load_checkpoint(tag="best")
        return loss_history, val_loss_history,best_step
    
    def fit_survival(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        
        
        data = node_lib.Dataset(self.args.dataset, random_state=self.args.seed,
                                X_train=np.array(X, dtype=np.float32), 
                                y_train=np.array(y, dtype=np.float32),
                                X_valid=np.array(X_val, dtype=np.float32), 
                                y_valid=np.array(y_val, dtype=np.float32))
        
        with torch.no_grad():
            
            _ = self.model(torch.as_tensor(data.X_train[:1000], device=self.device))
            
        experiment_name = f"{self.args.dataset}_survival_{cross_val_index}"

        
        log_dir = os.path.join('logs', experiment_name)
        if os.path.exists(log_dir):
            try:
                shutil.rmtree(log_dir)
            except Exception as e:
                print(f"Warning: Failed to delete {log_dir}. Error: {e}")
                
        
        loss_func_instance = NegativeLogLikelihood() 
        c_index_metric = CIndexMetric()              
        
        
        
        self.trainer = node_lib.Trainer(
            model=self.model, 
            loss_function=loss_func_instance,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(lr=1e-3, nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=False,
            n_last_checkpoints=5
        )
        
        
        best_c_index = -1.0  
        best_step = 0
        
        loss_history = []
        val_c_index_history = []

        
        early_stopping_steps = self.args.early_stopping_rounds * self.args.logging_period

        print(f"Starting Survival Training... (Early Stopping: {early_stopping_steps} steps)")
        
        
        for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, 
                                                  batch_size=self.args.batch_size, 
                                                  shuffle=True, 
                                                  epochs=self.args.epochs, 
                                                  seed=self.args.seed):
            
            # --- Train Step ---
            metrics = self.trainer.train_on_batch(*batch, device=self.device)
            loss_history.append(metrics['loss'].item())

            # --- Validation & Logging Step ---
            if self.trainer.step % self.args.logging_period == 0:
                
                
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')

                
                
                with torch.no_grad():
                    
                    X_val_tensor = torch.as_tensor(data.X_valid, device=self.device)
                    
                    
                    
                    val_risk_scores = self.model(X_val_tensor)
                    
                    
                    
                    
                    current_c_index = c_index_metric(data.y_valid, val_risk_scores)

                print("Step: %d, Train CoxLoss: %.5f, Val C-Index: %0.5f" % 
                      (self.trainer.step, metrics['loss'], current_c_index))
                
                val_c_index_history.append(current_c_index)

                
                if current_c_index > best_c_index:
                    best_c_index = current_c_index
                    best_step = self.trainer.step
                    self.trainer.save_checkpoint(tag='best') 
                    # print(f"New Best C-Index: {best_c_index:.5f} at step {best_step}")

                
                self.trainer.load_checkpoint() 
                self.trainer.remove_old_temp_checkpoints()

                
                if self.trainer.step > best_step + early_stopping_steps:
                    print('BREAK. No improvement for {} steps'.format(early_stopping_steps))
                    print("Best step: ", best_step)
                    print("Best Val C-Index: %0.5f" % best_c_index)
                    break

        
        self.trainer.load_checkpoint(tag="best")
        
        # =========================================================
        
        # =========================================================
        print("Fitting Breslow Estimator for NODE...")
        
        
        if not hasattr(self, 'breslow'):
            self.breslow = BreslowEstimator()

        
        self.model.eval() 
        risk_scores_list = []
        
        
        batch_size = self.args.batch_size
        num_samples = len(data.X_train)
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                
                X_batch_np = data.X_train[i : i + batch_size]
                X_batch = torch.as_tensor(X_batch_np, device=self.device)
                
                
                out = self.model(X_batch)
                
                
                risk_scores_list.append(out.cpu().numpy())
        
        
        log_risk_scores = np.concatenate(risk_scores_list, axis=0).flatten()
        
        
        train_times = data.y_train[:, 0]
        train_events = data.y_train[:, 1]
        
        
        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        
        print("Breslow Estimator fitted successfully.")
        # =========================================================

        return loss_history, val_c_index_history, best_step
        
        
    def predict(self, X):
        """
        NODE 模型的预测函数
        - Survival: 自行处理 (PyTorch推理 + Breslow计算)
        - 其他: 调用父类 (super().predict)
        """
        
        if self.args.objective == "Survival":           
            
            
            X_np = np.array(X, dtype=np.float32)
            self.model.eval()
            
            batch_size = self.args.batch_size
            num_samples = len(X_np)
            raw_outputs_list = []
            
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    
                    X_batch = torch.as_tensor(X_np[i : i + batch_size], device=self.device)
                    
                    out = self.model(X_batch)
                    
                    raw_outputs_list.append(out.cpu().numpy())
            
            
            
            log_risk_scores = np.concatenate(raw_outputs_list, axis=0).flatten()
            
            
            if hasattr(self, 'breslow'):
                
                time_point = 3652.0 
                
                
                surv_probs = self.breslow.predict_survival_probabilities(
                    log_risk_scores, time_point
                ).flatten()
                
                
                event_probs = 1.0 - surv_probs
            else:
                # print("Warning: Breslow Estimator not fitted in fit_survival. Returning zeros.")
                event_probs = np.zeros_like(log_risk_scores)
            
            
            self.predictions = log_risk_scores
            self.prediction_probabilities = event_probs
            
            
            return log_risk_scores, event_probs

        
        else:
            return super().predict(X)
        

    def fit_best_n_epochs(self, X, y, best_n_epochs):
        """
        使用确定的最佳步数(best_step)进行全量训练。
        注意：这里的 best_n_epochs 实际上对应 fit 函数返回的 best_step。
        """
        
        if self.args.objective == "Survival":
            print("Detected Survival Objective. Redirecting to fit_best_n_epochs_survival...")
            return self.fit_best_n_epochs_survival(X, y, best_n_epochs)
        
        
        target_step = int(best_n_epochs)

        
        
        data = node_lib.Dataset(self.args.dataset, random_state=self.args.seed,
                                X_train=np.array(X, dtype=np.float32), y_train=np.array(y, dtype=np.float32),
                                X_valid=None, y_valid=None)

        
        with torch.no_grad():
            
            init_data = torch.as_tensor(data.X_train[:1000], device=self.device)
            res = self.model(init_data)

        experiment_name = f"{self.args.dataset}_retrain"
        
        # -----------------------------------------------------------
        
        
        log_dir = os.path.join('logs', experiment_name)
        if os.path.exists(log_dir):
            print(f"Cleaning up previous experiment logs at: {log_dir}")
            try:
                shutil.rmtree(log_dir) 
            except Exception as e:
                print(f"Warning: Failed to delete {log_dir}. Error: {e}")
        # -----------------------------------------------------------

        
        if self.args.objective == "regression":
            loss_func = F.mse_loss
        elif self.args.objective == "classification":
            loss_func = F.cross_entropy
            data.y_train = data.y_train.astype(int)
        elif self.args.objective == "binary":
            loss_func = F.binary_cross_entropy_with_logits
            data.y_train = data.y_train.reshape(-1, 1)

        
        self.trainer = node_lib.Trainer(
            model=self.model, loss_function=loss_func,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(lr=1e-3, nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=False,
            n_last_checkpoints=5 
        )

        
        print(f"Retraining for specific steps: {target_step}")

        for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, batch_size=self.args.batch_size, shuffle=True,
                                                  epochs=self.args.epochs, seed=self.args.seed):
            
            
            self.trainer.train_on_batch(*batch, device=self.device)

            
            
            if self.trainer.step % self.args.logging_period == 0:
                self.trainer.save_checkpoint()
                self.trainer.remove_old_temp_checkpoints()

            
            if self.trainer.step >= target_step:
                print(f"Reached target step: {self.trainer.step}")
                
                
                
                
                self.trainer.save_checkpoint() 
                self.trainer.average_checkpoints(out_tag='final_best')
                self.trainer.load_checkpoint(tag='final_best')
                
                print("Loaded averaged weights. Retraining complete.")
                break
    
    def fit_best_n_epochs_survival(self, X, y, best_n_epochs):
        """
        生存分析专用的全量重训练函数
        """

        
        target_step = int(best_n_epochs)
        
        
        data = node_lib.Dataset(self.args.dataset, random_state=self.args.seed,
                                X_train=np.array(X, dtype=np.float32), 
                                y_train=np.array(y, dtype=np.float32),
                                X_valid=None, y_valid=None)

        
        with torch.no_grad():
            init_data = torch.as_tensor(data.X_train[:1000], device=self.device)
            _ = self.model(init_data)

        experiment_name = f"{self.args.dataset}_survival_retrain"

        
        log_dir = os.path.join('logs', experiment_name)
        if os.path.exists(log_dir):
            try:
                shutil.rmtree(log_dir)
            except Exception as e:
                print(f"Warning: Failed to delete {log_dir}. Error: {e}")

        
        loss_func_instance = NegativeLogLikelihood()

        
        self.trainer = node_lib.Trainer(
            model=self.model, 
            loss_function=loss_func_instance,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(lr=1e-3, nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=False,
            n_last_checkpoints=5
        )

        print(f"Retraining Survival Model for specific steps: {target_step}")

        
        
        for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, 
                                                  batch_size=self.args.batch_size, 
                                                  shuffle=True,
                                                  epochs=self.args.epochs,
                                                  seed=self.args.seed):
            
            
            self.trainer.train_on_batch(*batch, device=self.device)

            
            if self.trainer.step % self.args.logging_period == 0:
                self.trainer.save_checkpoint()
                self.trainer.remove_old_temp_checkpoints()

            
            if self.trainer.step >= target_step:
                print(f"Reached target step: {self.trainer.step}")
                
                
                self.trainer.save_checkpoint() 
                self.trainer.average_checkpoints(out_tag='final_best')
                self.trainer.load_checkpoint(tag='final_best')
                
                print("Loaded averaged weights.")
                break
        
        # =========================================================
        
        # =========================================================
        print("Re-fitting Breslow Estimator on full dataset...")
        
        if not hasattr(self, 'breslow'):
            self.breslow = BreslowEstimator()
        
        
        self.model.eval()
        risk_scores_list = []
        batch_size = self.args.batch_size
        num_samples = len(data.X_train)
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                X_batch_np = data.X_train[i : i + batch_size]
                X_batch = torch.as_tensor(X_batch_np, device=self.device)
                out = self.model(X_batch)
                risk_scores_list.append(out.cpu().numpy())
        
        log_risk_scores = np.concatenate(risk_scores_list, axis=0).flatten()
        
        train_times = data.y_train[:, 0]
        train_events = data.y_train[:, 1]
        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        
        print("Retraining Complete.")
        
        return
    
    def predict_helper(self, X):
        X_test = torch.as_tensor(np.array(X, dtype=float), device=self.device, dtype=torch.float32)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=self.args.val_batch_size)

            if self.args.objective == "classification":
                prediction = F.softmax(prediction, dim=1)
            elif self.args.objective == "binary":
                prediction = torch.sigmoid(prediction)

            prediction = check_numpy(prediction)

        self.predictions = prediction
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        print("Saving at", filename)
        self.trainer.save_checkpoint(path=filename)

    def load_model(self, filename_extension=""):
            """
            加载 NODE 模型权重。
            对应 save_checkpoint 保存的格式：{'model': state_dict, 'opt': ..., 'step': ...}
            """
            
            filename = get_output_path(self.args, directory="models", filename="m", 
                                    extension=filename_extension, file_type="pt")
            
            if not os.path.exists(filename):
                raise FileNotFoundError(f"No checkpoint found at {filename}")
                
            print(f"Loading NODE model from {filename}...")
            
            
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            
            
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            from collections import OrderedDict
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k
                
                if k.startswith('model.'):
                    name = k[6:] 
                new_state_dict[name] = v
            
            
            self.model.load_state_dict(new_state_dict)
                
            self.model.to(self.device)
            self.model.eval() 
            print("Model weights loaded successfully.")

    def fit_breslow(self, X_train, y_train):
            """
            基于加载的权重，重新在训练集上分批推理并拟合 Breslow Estimator。
            """
            if self.args.objective != "Survival":
                return

            print("Fitting Breslow Estimator for NODE...")
            
            
            if not hasattr(self, 'breslow'):
                self.breslow = BreslowEstimator()

            
            self.model.eval()
            risk_scores_list = []
            
            
            
            batch_size = getattr(self.args, 'val_batch_size', 1024) 
            num_samples = len(X_train)
            
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    
                    X_batch_np = X_train[i : i + batch_size]
                    
                    
                    X_batch = torch.as_tensor(X_batch_np, dtype=torch.float32, device=self.device)
                    
                    
                    out = self.model(X_batch)
                    
                    
                    risk_scores_list.append(out.cpu().numpy())
            
            
            log_risk_scores = np.concatenate(risk_scores_list, axis=0).flatten()
            
            
            train_times = y_train[:, 0]
            train_events = y_train[:, 1]
            
            
            self.breslow.fit(log_risk_scores, train_events, train_times)
            
            print("Breslow Estimator fitted successfully.")


    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_layers": trial.suggest_categorical("num_layers", [2,4]),
            "total_tree_count": trial.suggest_categorical("total_tree_count", [256,512,1024]),
            "tree_depth": trial.suggest_categorical("tree_depth", [2,4,6]),
            "tree_output_dim": trial.suggest_int("tree_output_dim", 1, 2)
            # "num_layers": trial.suggest_categorical("num_layers", [4]),
            # "total_tree_count": trial.suggest_categorical("total_tree_count", [1024]),
            # "tree_depth": trial.suggest_categorical("tree_depth", [6]),
            # "tree_output_dim": trial.suggest_int("tree_output_dim", 2, 2)
        }
        return params
