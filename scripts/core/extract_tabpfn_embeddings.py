"""Extract TabPFN embeddings using fold-local preprocessing."""

import os
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import gc

from utils.load_data import load_Survial_train_datas
from utils.parser import get_parser
from models.baseline_models import process_data

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding

# ==========================================

# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def safe_extract(extractor, X_tr, y_tr, X_tgt, source):
    
    if hasattr(X_tr, "to_numpy"): X_tr = X_tr.to_numpy()
    if hasattr(y_tr, "to_numpy"): y_tr = y_tr.to_numpy()
    if hasattr(X_tgt, "to_numpy"): X_tgt = X_tgt.to_numpy()
    
    if y_tr.ndim > 1: y_tr = y_tr.flatten()

    
    with torch.no_grad():
        out = extractor.get_embeddings(X_tr, y_tr, X_tgt, data_source=source)
        
        if isinstance(out, (list, tuple)):
            out = out[0]
            
        
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
            
        
        if out.ndim == 3:
            out = out.mean(axis=0)
            
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return out

# ==========================================

# ==========================================
def generate_and_save_embeddings(args):
    print(f"\n========== Start Embedding Pipeline for {args.disease_type} | {args.data_type} | {args.protein_source} ==========")
    
    seed_everything(args.seed)
    
    
    save_base_dir = os.path.join(
        str(args.artifact_dir), "tabpfn_embeddings",
        args.disease_type, 
        "classic" if args.data_type == "classic" else f"{args.data_type}_lassonet"
    )
    os.makedirs(save_base_dir, exist_ok=True)

    
    print("Loading raw data...")
    X_eur, y_eur, X_asian, y_asian, X_other, y_other, Cat_index, feature_names = load_Survial_train_datas(
        args, protein_source=args.protein_source
    )

    
    
    X_opt, X_holdout_eur, y_opt, y_holdout_eur = train_test_split(
        X_eur, y_eur, 
        test_size=args.ratio, 
        random_state=args.seed,
        stratify=y_eur[:, 1]
    )

    
    external_test_sets = {
        "holdout_eur": (X_holdout_eur, y_holdout_eur),
        "asian": (X_asian, y_asian),
        "other": (X_other, y_other)
    }

    
    for name, (_, y_ext) in external_test_sets.items():
        np.save(os.path.join(save_base_dir, f"y_{name}.npy"), y_ext)

    
    kf = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=args.seed)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_opt, y_opt[:, 1])):
        print(f"\n>>> Processing Fold {fold_idx + 1}/5 ...")
        fold_dir = os.path.join(save_base_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        
        X_train_fold, y_train_fold = X_opt[train_index], y_opt[train_index]
        X_val_fold, y_val_fold = X_opt[val_index], y_opt[val_index]

        # Fit categorical encoding and scaling on this training fold only.
        X_train_fold, transformer = process_data(X_train_fold, Cat_index, method="deep")
        X_val_fold = process_data(X_val_fold, Cat_index, transformer=transformer)
        X_holdout_fold = process_data(X_holdout_eur, Cat_index, transformer=transformer)
        X_asian_fold = process_data(X_asian, Cat_index, transformer=transformer)
        X_other_fold = process_data(X_other, Cat_index, transformer=transformer)

        
        np.save(os.path.join(fold_dir, "y_train.npy"), y_train_fold)
        np.save(os.path.join(fold_dir, "y_val.npy"), y_val_fold)

        
        y_cls_train = y_train_fold[:, 1].astype(int)  
        y_reg_train = np.log1p(y_train_fold[:, 0])    

        # ==================================================
        
        # ==================================================
        print("    - Initializing Classifier...")
        clf = TabPFNClassifier(n_estimators=4, device='cuda', random_state=args.seed, 
                            inference_precision=torch.float32, 
                            categorical_features_indices=Cat_index,
                            ignore_pretraining_limits=True)
        emb_clf = TabPFNEmbedding(tabpfn_clf=clf, n_fold=5)

        print("    - Extracting Classification Embeddings...")
        emb_cls_train = safe_extract(emb_clf, X_train_fold, y_cls_train, X_train_fold, "train")
        emb_cls_val   = safe_extract(emb_clf, X_train_fold, y_cls_train, X_val_fold, "test")
        e_cls_holdout = safe_extract(emb_clf, X_train_fold, y_cls_train, X_holdout_fold, "test")
        e_cls_asian   = safe_extract(emb_clf, X_train_fold, y_cls_train, X_asian_fold, "test")
        e_cls_other   = safe_extract(emb_clf, X_train_fold, y_cls_train, X_other_fold, "test")

        
        del clf, emb_clf
        gc.collect()                  
        torch.cuda.empty_cache()      
        # ==================================================


        # ==================================================
        
        # ==================================================
        print("    - Initializing Regressor...")
        reg = TabPFNRegressor(n_estimators=4, device='cuda', random_state=args.seed, 
                            inference_precision=torch.float32, 
                            categorical_features_indices=Cat_index,
                            ignore_pretraining_limits=True)
        emb_reg = TabPFNEmbedding(tabpfn_reg=reg, n_fold=5)

        print("    - Extracting Regression Embeddings...")
        emb_reg_train = safe_extract(emb_reg, X_train_fold, y_reg_train, X_train_fold, "train")
        emb_reg_val   = safe_extract(emb_reg, X_train_fold, y_reg_train, X_val_fold, "test")
        e_reg_holdout = safe_extract(emb_reg, X_train_fold, y_reg_train, X_holdout_fold, "test")
        e_reg_asian   = safe_extract(emb_reg, X_train_fold, y_reg_train, X_asian_fold, "test")
        e_reg_other   = safe_extract(emb_reg, X_train_fold, y_reg_train, X_other_fold, "test")

        
        del reg, emb_reg
        gc.collect()
        torch.cuda.empty_cache()
        # ==================================================

        # ==================================================
        
        # ==================================================
        print("    - Concatenating and saving...")
        emb_train_final = np.concatenate([emb_cls_train, emb_reg_train], axis=1)
        np.save(os.path.join(fold_dir, "X_train.npy"), emb_train_final)

        emb_val_final = np.concatenate([emb_cls_val, emb_reg_val], axis=1)
        np.save(os.path.join(fold_dir, "X_val.npy"), emb_val_final)

        
        e_holdout_final = np.concatenate([e_cls_holdout, e_reg_holdout], axis=1)
        np.save(os.path.join(fold_dir, "X_holdout_eur.npy"), e_holdout_final)

        e_asian_final = np.concatenate([e_cls_asian, e_reg_asian], axis=1)
        np.save(os.path.join(fold_dir, "X_asian.npy"), e_asian_final)

        e_other_final = np.concatenate([e_cls_other, e_reg_other], axis=1)
        np.save(os.path.join(fold_dir, "X_other.npy"), e_other_final)

    print(f"✅ Embeddings saved successfully at: {save_base_dir}\n")

