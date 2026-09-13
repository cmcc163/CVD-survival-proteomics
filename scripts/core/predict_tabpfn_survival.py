"""Generate fold-ensemble survival predictions from TabPFN embeddings."""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import optuna
import logging


from models import str2model
from utils.parser import get_parser
from utils.io_utils import save_results_to_file
from utils.paths import optuna_storage

# =============================================================================

# =============================================================================
def save_tabpfn_predictions_to_csv(args, datasets):
    """
    保存预测结果到 CSV
    datasets: 字典 {'val': (y_true, scores, probs), ...}
    """
    print("=" * 30)
    print("Saving TabPFN predictions to CSV...")
    
    
    save_dir = os.path.join(str(args.artifact_dir), "predictions", "TabPFN", args.disease_type)
    os.makedirs(save_dir, exist_ok=True)
    
    
    if args.data_type == 'classic':
        config_name = "classic"
    else:
        config_name = f"{args.data_type}_lassonet"

    
    for split_name, (y_data, scores, probs) in datasets.items():
        if y_data is None:
            continue

        if isinstance(y_data, np.ndarray):
            time_arr = y_data[:, 0]
            event_arr = y_data[:, 1]
        else:
            print(f"Warning: y_data type {type(y_data)} not supported for {split_name}")
            continue

        df = pd.DataFrame({
            'time': time_arr,
            'event': event_arr.astype(int),
            'risk_score': scores,
            'prob_10yr': probs
        })
        
        filename = f"{config_name}_{split_name}.csv"
        file_path = os.path.join(save_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

    print("=" * 30)


# =============================================================================

# =============================================================================
def run_inference_with_best_params(args):
    print(f"\n>>> Processing: {args.disease_type} | {args.data_type} | {args.protein_source}")

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    config_name = "classic" if args.data_type == "classic" else f"{args.data_type}_lassonet"
    data_path = os.path.join(str(args.artifact_dir), "tabpfn_embeddings", args.disease_type, config_name)
    
    args.datas = data_path
    if not os.path.exists(data_path):
        print(f"❌ Error: Data path does not exist: {data_path}")
        return

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    
    if args.data_type == 'classic':
        study_name = args.model_name + "_" + args.dataset + "_" + args.disease_type + "_" + args.data_type
    else:
        study_name = args.model_name + "_" + args.dataset + "_" + args.disease_type + "_" + args.data_type + "_" + args.protein_source
    
    db_url = optuna_storage(args, study_name)
    
    print(f"Loading Optuna study: {study_name}")
    
    try:
        
        study = optuna.load_study(study_name=study_name, storage=db_url)
        best_params = study.best_trial.params
        print(f"✅ Best Params Loaded: {best_params}")
        
        
        if "best_n_epochs" in study.best_trial.user_attrs:
            args.best_n_epochs = study.best_trial.user_attrs["best_n_epochs"]
            print(f"   Best Epochs: {args.best_n_epochs}")
    except Exception as e:
        print(f"❌ Error loading study from DB: {e}")
        return

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    try:
        y_val_true = np.load(os.path.join(args.datas, 'y_holdout_eur.npy'))
        y_asian_true = np.load(os.path.join(args.datas, 'y_asian.npy'))
        y_other_true = np.load(os.path.join(args.datas, 'y_other.npy'))
    except FileNotFoundError as e:
        print(f"❌ Error loading ground truth files: {e}")
        return

    
    ensemble_probs_val = np.zeros(len(y_val_true))
    ensemble_probs_asian = np.zeros(len(y_asian_true))
    ensemble_probs_other = np.zeros(len(y_other_true))
    
    ensemble_scores_val = np.zeros(len(y_val_true))
    ensemble_scores_asian = np.zeros(len(y_asian_true))
    ensemble_scores_other = np.zeros(len(y_other_true))

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    
    args.num_features = 384
    
    ModelClass = str2model(args.model_name)
    
    
    model = ModelClass(best_params, args)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # args.num_splits
    for i in range(args.num_splits):
        fold_name = f"best_fold_{i}"
        print(f"  - Ensemble Fold {i}/{args.num_splits}: Loading {fold_name}...")
        
        
        try:
            X_train = np.load(os.path.join(args.datas, f'fold_{i}', 'X_train.npy'))
            y_train = np.load(os.path.join(args.datas, f'fold_{i}', 'y_train.npy'))
            
            X_val = np.load(os.path.join(args.datas, f'fold_{i}', 'X_holdout_eur.npy'))
            X_asian = np.load(os.path.join(args.datas, f'fold_{i}', 'X_asian.npy'))
            X_other = np.load(os.path.join(args.datas, f'fold_{i}', 'X_other.npy'))
        except FileNotFoundError:
            print(f"    ⚠️ Warning: Fold {i} data missing in {args.datas}. Skipping.")
            continue

        
        try:
            model.load_model(fold_name)
        except Exception as e:
            print(f"    ⚠️ Error loading model weight: {e}")
            continue
        
        
        model.fit_breslow(X_train, y_train)
        
        
        s_val, p_val = model.predict(X_val)
        s_asian, p_asian = model.predict(X_asian)
        s_other, p_other = model.predict(X_other)
        
        
        ensemble_scores_val += s_val
        ensemble_probs_val += p_val
        ensemble_scores_asian += s_asian
        ensemble_probs_asian += p_asian
        ensemble_scores_other += s_other
        ensemble_probs_other += p_other

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    avg_scores_val = ensemble_scores_val / args.num_splits
    avg_probs_val = ensemble_probs_val / args.num_splits
    
    avg_scores_asian = ensemble_scores_asian / args.num_splits
    avg_probs_asian = ensemble_probs_asian / args.num_splits
    
    avg_scores_other = ensemble_scores_other / args.num_splits
    avg_probs_other = ensemble_probs_other / args.num_splits

    datasets_to_save = {
        'val': (y_val_true, avg_scores_val, avg_probs_val),
        'asian': (y_asian_true, avg_scores_asian, avg_probs_asian),
        'other': (y_other_true, avg_scores_other, avg_probs_other)
    }

    save_tabpfn_predictions_to_csv(args, datasets_to_save)


# =============================================================================

# =============================================================================

