# The SAINT model.
from models.basemodel_torch import BaseModelTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.survival_tools import NegativeLogLikelihood,CIndexMetric
from utils.breslow import BreslowEstimator
import numpy as np
from torch import einsum
from einops import rearrange

from models.saint_lib.models.pretrainmodel import SAINT as SAINTModel
from models.saint_lib.data_openml import DataSetCatCon
from models.saint_lib.augmentations import embed_data_mask
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
    SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
    (https://arxiv.org/abs/2106.01342)
    
    Code adapted from: https://github.com/somepago/saint
'''

class AttentionSurvivalModel(BaseModelTorch):
    """Shared survival implementation for SAINT and FT-Transformer.

    Public subclasses set ``attention_type`` explicitly, preventing model names
    from silently changing the architecture inside a shared constructor.
    """

    attention_type = "colrow"

    def __init__(self, params, args):
        super().__init__(params, args)
        seed_everything(args.seed)
        if args.cat_idx:
            num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            cat_dims = np.append(np.array([1]), np.array(args.cat_dims)).astype(int)
        else:
            num_idx = list(range(args.num_features))
            cat_dims = np.array([1])

        # Decreasing some hyperparameter to cope with memory issues
        dim = self.params["dim"]
        self.batch_size = self.args.batch_size

        print("Using dim %d and batch size %d" % (dim, self.batch_size))
        attentiontype = self.attention_type
        self.model = SAINTModel(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=dim,
            dim_out=1,
            depth=self.params["depth"],  # 6
            heads=self.params["heads"],  # 8
            attn_dropout=0.1,  # 0.1
            ff_dropout=self.params["dropout"],  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            # attentiontype="col",
            # attentiontype="row",
            attentiontype=attentiontype,
            final_mlp_style="sep",
            y_dim=args.num_classes,
            mlp_hiddle=100,
            final_hiddle=1000
        )

        if self.args.data_parallel:
            self.model.transformer = nn.DataParallel(self.model.transformer, device_ids=self.args.gpu_ids)
            self.model.mlpfory = nn.DataParallel(self.model.mlpfory, device_ids=self.args.gpu_ids)

    # def fit(self, X, y, X_val=None, y_val=None):

        # if self.args.objective == 'binary':
        #     criterion = nn.BCEWithLogitsLoss()
        # elif self.args.objective == 'classification':
        #     criterion = nn.CrossEntropyLoss()
        # else:
        #     criterion = nn.MSELoss()

        # optimizer = optim.AdamW(self.model.parameters(), lr=0.00003)

        # self.model.to(self.device)

        # # SAINT wants it like this...
        # X = {'data': X, 'mask': np.ones_like(X)}
        # y = {'data': y.reshape(-1, 1)}
        # X_val = {'data': X_val, 'mask': np.ones_like(X_val)}
        # y_val = {'data': y_val.reshape(-1, 1)}

        # train_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        # trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,pin_memory=True)

        # val_ds = DataSetCatCon(X_val, y_val, self.args.cat_idx, self.args.objective)
        # valloader = DataLoader(val_ds, batch_size=self.args.val_batch_size, shuffle=True,pin_memory=True)

        # min_val_loss = float("inf")
        # min_val_loss_idx = 0

        # loss_history = []
        # val_loss_history = []

        # for epoch in range(self.args.epochs):
        #     self.model.train()

        #     for i, data in enumerate(trainloader, 0):
        #         optimizer.zero_grad()

        #         # x_categ is the the categorical data,
        #         # x_cont has continuous data,
        #         # y_gts has ground truth ys.
        #         # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
        #         # token) set to 0s.
        #         # con_mask is an array of ones same shape as x_cont.
        #         x_categ, x_cont, y_gts, cat_mask, con_mask = data

        #         x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
        #         cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

        #         # We are converting the data to embeddings in the next step
        #         _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

        #         reps = self.model.transformer(x_categ_enc, x_cont_enc)

        #         # select only the representations corresponding to CLS token
        #         # and apply mlp on it in the next step to get the predictions.
        #         y_reps = reps[:, 0, :]

        #         y_outs = self.model.mlpfory(y_reps)

        #         if self.args.objective == "regression":
        #             y_gts = y_gts.to(self.device)
        #         elif self.args.objective == "classification":
        #             y_gts = y_gts.to(self.device).squeeze()
        #         else:
        #             y_gts = y_gts.to(self.device).float()

        #         loss = criterion(y_outs, y_gts)
        #         loss.backward()
        #         optimizer.step()

        #         loss_history.append(loss.item())

        #         # print("Loss", loss.item())

        #     # Early Stopping
        #     val_loss = 0.0
        #     val_dim = 0
        #     self.model.eval()
        #     with torch.no_grad():
        #         for data in valloader:
        #             x_categ, x_cont, y_gts, cat_mask, con_mask = data

        #             x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
        #             cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

        #             _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
        #             reps = self.model.transformer(x_categ_enc, x_cont_enc)
        #             y_reps = reps[:, 0, :]
        #             y_outs = self.model.mlpfory(y_reps)

        #             if self.args.objective == "regression":
        #                 y_gts = y_gts.to(self.device)
        #             elif self.args.objective == "classification":
        #                 y_gts = y_gts.to(self.device).squeeze()
        #             else:
        #                 y_gts = y_gts.to(self.device).float()

        #             val_loss += criterion(y_outs, y_gts)
        #             val_dim += 1
        #     val_loss /= val_dim

        #     val_loss_history.append(val_loss.item())

        #     print("Epoch", epoch, "loss", val_loss.item())

        #     if val_loss < min_val_loss:
        #         min_val_loss = val_loss
        #         min_val_loss_idx = epoch

        #         # Save the currently best model
        #         self.save_model(filename_extension="best", directory="tmp")

        #     if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
        #         print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
        #         print("Early stopping applies.")
        #         break

        # self.load_model(filename_extension="best", directory="tmp")
        # return loss_history, val_loss_history
    
    def fit(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        """
        训练模型的主循环函数
        """
        if self.args.objective == "Survival":
            print("Detected Survival Objective. Redirecting to fit_survival...")
            return self.fit_survival(X, y, X_val, y_val, cross_val_index)
        if self.args.objective == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.objective == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)
        self.model.to(self.device)

        
        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': y.reshape(-1, 1)}
        if X_val is not None:
            X_val = {'data': X_val, 'mask': np.ones_like(X_val)}
            y_val = {'data': y_val.reshape(-1, 1)}
        train_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective) 
        trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        valloader = None
        if X_val is not None:
            val_ds = DataSetCatCon(X_val, y_val, self.args.cat_idx, self.args.objective)
            valloader = DataLoader(val_ds, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0
        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                x_categ, x_cont, y_gts, _, _ = data

                x_categ = x_categ.to(self.device)
                x_cont = x_cont.to(self.device)

                
                
                
                
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)

                
                n_cont = self.model.num_continuous
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                pos = torch.arange(x_input_enc.shape[1], device=self.device)
                x_input_enc += self.model.pos_encodings(pos)

                
                n_categ_total = self.model.num_categories
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                # ----------------------------------------------------

                y_outs = self.model(x_categ_enc_final, x_cont_enc_final)

                if self.args.objective == "regression":
                    y_gts = y_gts.to(self.device)
                elif self.args.objective == "classification":
                    y_gts = y_gts.to(self.device).squeeze().long()
                else:
                    y_gts = y_gts.to(self.device).float()

                loss = criterion(y_outs, y_gts)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(trainloader)
            loss_history.append(avg_epoch_loss)

            # ==================== Validation ====================
            if valloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for data in valloader:
                        x_categ, x_cont, y_gts, _, _ = data
                        x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)

                        
                        x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                        x_categ_enc = self.model.embeds(x_categ)
                        
                        x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                        for idx in range(n_cont):
                            x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                        x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                        x_input_enc += self.model.pos_encodings(torch.arange(x_input_enc.shape[1], device=self.device))
                        
                        x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                        x_cont_enc_final = x_input_enc[:, n_categ_total:, :]

                        y_outs = self.model(x_categ_enc_final, x_cont_enc_final)

                        if self.args.objective == "regression":
                            y_gts = y_gts.to(self.device)
                        elif self.args.objective == "classification":
                            y_gts = y_gts.to(self.device).squeeze().long()
                        else:
                            y_gts = y_gts.to(self.device).float()

                        val_loss += criterion(y_outs, y_gts).item()
                        val_steps += 1

                avg_val_loss = val_loss / val_steps
                val_loss_history.append(avg_val_loss)
                print(f"Epoch {epoch} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    min_val_loss_idx = epoch
                    self.save_model(filename_extension="best", directory="tmp")
                
                if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch} | Train Loss: {avg_epoch_loss:.4f}")

        if valloader is not None:
            self.load_model(filename_extension="best", directory="tmp")
            
        return loss_history, val_loss_history, min_val_loss_idx
    
    def fit_survival(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        """
        专用于生存分析的训练循环
        """
        
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0001) 
        self.model.to(self.device)

        
        criterion = NegativeLogLikelihood()
        c_index_metric = CIndexMetric()

        
        
        X_train_dict = {'data': X, 'mask': np.ones_like(X)}
        y_train_dict = {'data': y.reshape(-1, 2)} # Time, Event
        
        train_ds = DataSetCatCon(X_train_dict, y_train_dict, self.args.cat_idx, self.args.objective)
        trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        valloader = None
        if X_val is not None:
            X_val_dict = {'data': X_val, 'mask': np.ones_like(X_val)}
            y_val_dict = {'data': y_val.reshape(-1, 2)}
            val_ds = DataSetCatCon(X_val_dict, y_val_dict, self.args.cat_idx, self.args.objective)
            
            valloader = DataLoader(val_ds, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=True)

        
        best_val_c_index = -1.0
        best_epoch = 0
        loss_history = []
        val_c_index_history = []
        
        
        n_cont = self.model.num_continuous
        n_categ_total = self.model.num_categories

        print(f"Starting Survival Training... (Total Epochs: {self.args.epochs})")

        
        for epoch in range(self.args.epochs):
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                
                x_categ, x_cont, y_gts, _, _ = data

                x_categ = x_categ.to(self.device)
                x_cont = x_cont.to(self.device)
                
                y_gts = y_gts.to(self.device).float()

                
                
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)

                
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                pos = torch.arange(x_input_enc.shape[1], device=self.device)
                x_input_enc += self.model.pos_encodings(pos)

                
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                # --------------------------------------------------------------------

                # Forward pass -> Risk Score (Log Hazard)
                y_outs = self.model(x_categ_enc_final, x_cont_enc_final)

                
                loss = criterion(y_outs, y_gts)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(trainloader)
            loss_history.append(avg_epoch_loss)

            # ==================== Validation ====================
            if valloader is not None:
                self.model.eval()
                val_preds_list = []
                val_targets_list = []

                with torch.no_grad():
                    for data in valloader:
                        x_categ, x_cont, y_gts, _, _ = data
                        x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                        
                        
                        x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                        x_categ_enc = self.model.embeds(x_categ)
                        x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                        for idx in range(n_cont):
                            x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))
                        x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                        x_input_enc += self.model.pos_encodings(torch.arange(x_input_enc.shape[1], device=self.device))
                        x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                        x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                        # ---------------------------

                        y_outs = self.model(x_categ_enc_final, x_cont_enc_final)
                        
                        
                        val_preds_list.append(y_outs.cpu()) # Risk scores
                        val_targets_list.append(y_gts.cpu()) # (Time, Event)

                
                val_preds = torch.cat(val_preds_list, dim=0)
                val_targets = torch.cat(val_targets_list, dim=0)
                
                
                current_c_index = c_index_metric(val_targets, val_preds)
                val_c_index_history.append(current_c_index)

                print(f"Epoch {epoch} | Train CoxLoss: {avg_epoch_loss:.5f} | Val C-Index: {current_c_index:.5f}")

                
                if current_c_index > best_val_c_index:
                    best_val_c_index = current_c_index
                    best_epoch = epoch
                    self.save_model(filename_extension="best", directory="tmp")
                
                if best_epoch + self.args.early_stopping_rounds < epoch:
                    print(f"Early stopping at epoch {epoch}. Best C-Index: {best_val_c_index:.5f}")
                    break
            else:
                print(f"Epoch {epoch} | Train CoxLoss: {avg_epoch_loss:.5f}")

        
        if valloader is not None:
            self.load_model(filename_extension="best", directory="tmp")
        
        # =========================================================
        
        # =========================================================
        print("Fitting Breslow Estimator for SAINT...")
        
        if not hasattr(self, 'breslow'):
            self.breslow = BreslowEstimator()

        self.model.eval()
        train_risk_scores_list = []

        
        
        
        breslow_loader = DataLoader(train_ds, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=True)
        
        
        train_times = []
        train_events = []

        with torch.no_grad():
            for data in breslow_loader:
                x_categ, x_cont, y_gts, _, _ = data
                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                
                
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                x_input_enc += self.model.pos_encodings(torch.arange(x_input_enc.shape[1], device=self.device))
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                # ---------------------

                out = self.model(x_categ_enc_final, x_cont_enc_final)
                train_risk_scores_list.append(out.cpu().numpy())
                
                
                train_times.append(y_gts[:, 0].numpy())
                train_events.append(y_gts[:, 1].numpy())

        
        log_risk_scores = np.concatenate(train_risk_scores_list, axis=0).flatten()
        train_times = np.concatenate(train_times, axis=0)
        train_events = np.concatenate(train_events, axis=0)

        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        print("Breslow Estimator fitted successfully.")

        return loss_history, val_c_index_history, best_epoch
    
    def fit_breslow(self, X_train, y_train):
        """
        基于加载的权重，重新在训练集上分批推理并拟合 Breslow Estimator。
        适配 SAINT 的数据字典输入和 Embedding 处理逻辑。
        """
        if self.args.objective != "Survival":
            return

        print("Fitting Breslow Estimator for SAINT...")

        
        if not hasattr(self, 'breslow'):
            self.breslow = BreslowEstimator()

        
        
        X_train_dict = {'data': X_train, 'mask': np.ones_like(X_train)}
        y_train_dict = {'data': y_train.reshape(-1, 2)}
        
        
        train_ds = DataSetCatCon(X_train_dict, y_train_dict, self.args.cat_idx, self.args.objective)
        
        batch_size = getattr(self.args, 'val_batch_size', 1024)
        breslow_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

        
        self.model.eval()
        
        risk_scores_list = []
        train_times = []
        train_events = []

        
        n_cont = self.model.num_continuous
        n_categ_total = self.model.num_categories

        
        with torch.no_grad():
            for data in breslow_loader:
                x_categ, x_cont, y_gts, _, _ = data
                x_categ = x_categ.to(self.device)
                x_cont = x_cont.to(self.device)

                
                
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)

                
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                pos = torch.arange(x_input_enc.shape[1], device=self.device)
                x_input_enc += self.model.pos_encodings(pos)

                
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]

                
                out = self.model(x_categ_enc_final, x_cont_enc_final)
                
                
                risk_scores_list.append(out.cpu().numpy())
                train_times.append(y_gts[:, 0].numpy())
                train_events.append(y_gts[:, 1].numpy())

        
        log_risk_scores = np.concatenate(risk_scores_list, axis=0).flatten()
        train_times = np.concatenate(train_times, axis=0)
        train_events = np.concatenate(train_events, axis=0)

        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        
        print(f"Breslow Estimator fitted successfully on {len(log_risk_scores)} samples.")
        
    def fit_best_n_epochs(self, X, y, best_n_epochs):
        """
        使用确定的最佳 Epoch 数进行全量训练。
        不划分验证集，不进行 Early Stopping。
        建议传入的 X, y 为合并了 Train 和 Val 的全量数据。
        """
        
        if self.args.objective == "Survival":
            print(f"Detected Survival Objective. Retraining on full dataset for {best_n_epochs} epochs...")
            return self.fit_best_n_epochs_survival(X, y, best_n_epochs)
        
        
        if self.args.objective == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.objective == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0003)
        self.model.to(self.device)

        
        
        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': y.reshape(-1, 1)}
        
        
        train_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        loss_history = []
        
        
        for epoch in range(int(best_n_epochs)):
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                x_categ, x_cont, y_gts, _, _ = data

                x_categ = x_categ.to(self.device)
                x_cont = x_cont.to(self.device)

                
                
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)

                
                n_cont = self.model.num_continuous
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                pos = torch.arange(x_input_enc.shape[1], device=self.device)
                x_input_enc += self.model.pos_encodings(pos)

                
                n_categ_total = self.model.num_categories
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                # ==============================================================================

                y_outs = self.model(x_categ_enc_final, x_cont_enc_final)

                
                if self.args.objective == "regression":
                    y_gts = y_gts.to(self.device)
                elif self.args.objective == "classification":
                    y_gts = y_gts.to(self.device).squeeze().long()
                else:
                    y_gts = y_gts.to(self.device).float()

                loss = criterion(y_outs, y_gts)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(trainloader)
            loss_history.append(avg_epoch_loss)
            
            
            print(f"Refit Epoch {epoch+1}/{int(best_n_epochs)} | Train Loss: {avg_epoch_loss:.4f}")

        return loss_history
    
    def fit_best_n_epochs_survival(self, X, y, best_n_epochs):
        """
        生存分析专用的全量重训练函数
        """
        
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0003)
        self.model.to(self.device)
        
        
        criterion = NegativeLogLikelihood()

        
        X_train_dict = {'data': X, 'mask': np.ones_like(X)}
        y_train_dict = {'data': y.reshape(-1, 2)} # Ensure (N, 2)
        
        train_ds = DataSetCatCon(X_train_dict, y_train_dict, self.args.cat_idx, self.args.objective)
        
        trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        
        n_cont = self.model.num_continuous
        n_categ_total = self.model.num_categories
        target_epochs = int(best_n_epochs)

        print(f"Starting Refit on full dataset (N={len(X)}) for {target_epochs} epochs.")

        
        for epoch in range(target_epochs):
            self.model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()
                x_categ, x_cont, y_gts, _, _ = data

                x_categ = x_categ.to(self.device)
                x_cont = x_cont.to(self.device)
                y_gts = y_gts.to(self.device).float()

                
                # 1. Categ
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)

                # 2. Cont
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                # 3. Cat & Pos
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                pos = torch.arange(x_input_enc.shape[1], device=self.device)
                x_input_enc += self.model.pos_encodings(pos)

                # 4. Split
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                # ====================================================================

                # Forward & Backward
                y_outs = self.model(x_categ_enc_final, x_cont_enc_final)
                loss = criterion(y_outs, y_gts)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch + 1) % 5 == 0 or (epoch + 1) == target_epochs:
                print(f"Refit Epoch {epoch+1}/{target_epochs} | Loss: {running_loss / len(trainloader):.5f}")

        # =========================================================
        
        # =========================================================
        print("Re-fitting Breslow Estimator on FULL dataset...")
        
        if not hasattr(self, 'breslow'):
            self.breslow = BreslowEstimator()

        self.model.eval()
        
        
        inference_loader = DataLoader(train_ds, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=True)
        
        train_risk_scores_list = []
        train_times = []
        train_events = []

        with torch.no_grad():
            for data in inference_loader:
                x_categ, x_cont, y_gts, _, _ = data
                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                
                # --- Embedding ---
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                x_input_enc += self.model.pos_encodings(torch.arange(x_input_enc.shape[1], device=self.device))
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                # -----------------

                out = self.model(x_categ_enc_final, x_cont_enc_final)
                train_risk_scores_list.append(out.cpu().numpy())
                
                
                train_times.append(y_gts[:, 0].numpy())
                train_events.append(y_gts[:, 1].numpy())

        
        log_risk_scores = np.concatenate(train_risk_scores_list, axis=0).flatten()
        train_times = np.concatenate(train_times, axis=0)
        train_events = np.concatenate(train_events, axis=0)

        
        self.breslow.fit(log_risk_scores, train_events, train_times)
        
        print("Refit Complete. Model is ready for inference.")
    
    
    def predict(self, X):
        """
        SAINT 模型的预测函数 (适配 Survival 任务)
        """
        
        if self.args.objective == "Survival":
            self.model.eval()
            
            
            
            X_dict = {'data': X, 'mask': np.ones_like(X)}
            y_dummy = {'data': np.zeros((len(X), 2))} 
            
            
            
            pred_ds = DataSetCatCon(X_dict, y_dummy, self.args.cat_idx, self.args.objective)
            pred_loader = DataLoader(pred_ds, batch_size=self.args.val_batch_size, shuffle=False, pin_memory=True)
            
            raw_outputs_list = []
            
            
            n_cont = self.model.num_continuous
            n_categ_total = self.model.num_categories

            with torch.no_grad():
                for data in pred_loader:
                    
                    
                    x_categ, x_cont, _, _, _ = data
                    
                    x_categ = x_categ.to(self.device)
                    x_cont = x_cont.to(self.device)

                    # ==========================================================
                    
                    # ==========================================================
                    
                    
                    
                    x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                    x_categ_enc = self.model.embeds(x_categ)

                    
                    x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                    for idx in range(n_cont):
                        x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                    
                    x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                    pos = torch.arange(x_input_enc.shape[1], device=self.device)
                    x_input_enc += self.model.pos_encodings(pos)

                    
                    x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                    x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                    
                    # ==========================================================
                    
                    
                    out = self.model(x_categ_enc_final, x_cont_enc_final)
                    raw_outputs_list.append(out.cpu().numpy())

            
            log_risk_scores = np.concatenate(raw_outputs_list, axis=0).flatten()

            
            if hasattr(self, 'breslow'):
                
                time_point = 3652.0 
                
                
                surv_probs = self.breslow.predict_survival_probabilities(
                    log_risk_scores, time_point
                ).flatten()
                
                # P(Event) = 1 - S(t)
                event_probs = 1.0 - surv_probs
            else:
                print("Warning: Breslow Estimator not fitted. Returning zeros.")
                event_probs = np.zeros_like(log_risk_scores)

            
            self.predictions = log_risk_scores
            self.prediction_probabilities = event_probs

            return log_risk_scores, event_probs

        
        else:
            return super().predict(X)
    
    def predict_helper(self, X):
        
        X = {'data': X, 'mask': np.ones_like(X)}
        
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, 
                                shuffle=False, num_workers=0, pin_memory=True)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data in testloader:
                
                x_categ, x_cont, _, _, _ = data

                x_categ = x_categ.to(self.device)
                x_cont = x_cont.to(self.device)

                
                
                
                
                x_categ = x_categ + self.model.categories_offset.type_as(x_categ)
                x_categ_enc = self.model.embeds(x_categ)

                
                n_cont = self.model.num_continuous
                x_cont_enc = torch.empty(x_cont.size(0), n_cont, self.model.dim, device=self.device)
                for idx in range(n_cont):
                    x_cont_enc[:, idx, :] = self.model.simple_MLP[idx](x_cont[:, idx].view(-1, 1))

                
                x_input_enc = torch.cat([x_categ_enc, x_cont_enc], dim=1)
                pos = torch.arange(x_input_enc.shape[1], device=self.device)
                x_input_enc += self.model.pos_encodings(pos)

                
                n_categ_total = self.model.num_categories
                x_categ_enc_final = x_input_enc[:, :n_categ_total, :]
                x_cont_enc_final = x_input_enc[:, n_categ_total:, :]
                
                # ==============================================================================

                
                y_outs = self.model(x_categ_enc_final, x_cont_enc_final)

                
                if self.args.objective == "binary":
                    
                    y_outs = torch.sigmoid(y_outs)
                elif self.args.objective == "classification":
                    
                    y_outs = F.softmax(y_outs, dim=1)
                

                predictions.append(y_outs.detach().cpu().numpy())

        return np.concatenate(predictions)
    
    # def predict_helper(self, X):
    #     X = {'data': X, 'mask': np.ones_like(X)}
    #     y = {'data': np.ones((X['data'].shape[0], 1))}

    #     test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
    #     testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

    #     self.model.eval()

    #     predictions = []

    #     with torch.no_grad():
    #         for data in testloader:
    #             x_categ, x_cont, y_gts, cat_mask, con_mask = data

    #             x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
    #             cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

    #             _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
    #             reps = self.model.transformer(x_categ_enc, x_cont_enc)
    #             y_reps = reps[:, 0, :]
    #             y_outs = self.model.mlpfory(y_reps)

    #             if self.args.objective == "binary":
    #                 y_outs = torch.sigmoid(y_outs)
    #             elif self.args.objective == "classification":
    #                 y_outs = F.softmax(y_outs, dim=1)

    #             predictions.append(y_outs.detach().cpu().numpy())
    #     return np.concatenate(predictions)

    def attribute(self, X, y, strategy=""):
        """ Generate feature attributions for the model input.
            Two strategies are supported: default ("") or "diag". The default strategie takes the sum
            over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
            of the attention map.
            return array with the same shape as X.
        """
        global my_attention
        # self.load_model(filename_extension="best", directory="tmp")

        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()
        # print(self.model)
        # Apply hook.
        my_attention = torch.zeros(0)

        def sample_attribution(layer, minput, output):
            global my_attention
            # print(minput)
            """ an hook to extract the attention maps. """
            h = layer.heads
            q, k, v = layer.to_qkv(minput[0]).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * layer.scale
            my_attention = sim.softmax(dim=-1)

        # print(type(self.model.transformer.layers[0][0].fn.fn))
        self.model.transformer.layers[0][0].fn.fn.register_forward_hook(sample_attribution)
        attributions = []
        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                # print(x_categ.shape, x_cont.shape)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                # y_reps = reps[:, 0, :]
                # y_outs = self.model.mlpfory(y_reps)
                if strategy == "diag":
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].diagonal(0, 1, 2))
                else:
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].sum(dim=1))

        attributions = np.concatenate(attributions)
        return attributions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            # "dim": trial.suggest_categorical("dim", [4,8,16,32]),
            # "depth": trial.suggest_categorical("depth", [1,2,4,6]),
            # "heads": trial.suggest_categorical("heads", [1,2,4]),
            # "dropout": trial.suggest_float("dropout", 0.1, 0.5)
            "dim": trial.suggest_categorical("dim", [8,16,32,64]),
            "depth": trial.suggest_categorical("depth", [1]),
            "heads": trial.suggest_categorical("heads", [2,4,8]),
            "dropout": trial.suggest_float("dropout", 0.5, 0.8)
        }
        return params
