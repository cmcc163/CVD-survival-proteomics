"""Generate development, hold-out, and cross-ancestry model predictions."""

import logging
import sys
import optuna
from models.baseline_models import process_data
from models import str2model
from utils.load_data import load_Survial_train_datas, set_category_dimensions
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import (
    get_output_path,
    load_preprocessor_from_file,
    save_hyperparameters_to_file,
    save_results_to_file,
)
from utils.parser import get_parser, get_given_parameters_parser
from utils.paths import optuna_storage
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
from sklearn.utils import resample
import numpy as np
from sksurv.util import Surv
import shutil
import os,glob
import pandas as pd

def generate_and_save_predictions(model, 
                                  X_train, y_train, 
                                  X_val, y_val, 
                                  X_asian, y_asian, 
                                  X_other, y_other, 
                                  args, n_iterations=1000):
    print(f"Generating and saving predictions for {args.model_name}...")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("Loading fold-specific preprocessors...")

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    ensemble_scores_train = np.zeros(len(y_train))
    ensemble_scores_val = np.zeros(len(y_val))
    ensemble_scores_asian = np.zeros(len(y_asian))
    ensemble_scores_other = np.zeros(len(y_other))

    
    ensemble_probs_10_train = np.zeros(len(y_train))
    ensemble_probs_10_val = np.zeros(len(y_val))
    ensemble_probs_10_asian = np.zeros(len(y_asian))
    ensemble_probs_10_other = np.zeros(len(y_other))

    
    ensemble_probs_15_train = np.zeros(len(y_train))
    ensemble_probs_15_val = np.zeros(len(y_val))
    ensemble_probs_15_asian = np.zeros(len(y_asian))
    ensemble_probs_15_other = np.zeros(len(y_other))
    
    ensemble_probs_20_train = np.zeros(len(y_train))    
    ensemble_probs_20_val = np.zeros(len(y_val))
    ensemble_probs_20_asian = np.zeros(len(y_asian))
    ensemble_probs_20_other = np.zeros(len(y_other))

    
    time_point_15yr = 15 * 365.2 
    time_point_20yr = 20 * 365.2 

    
    
    def safe_get_probs(model_obj, X_features, risk_scores, t):
        
        if hasattr(model_obj, 'breslow') and model_obj.breslow is not None:
            surv_probs = model_obj.breslow.predict_survival_probabilities(risk_scores, t).flatten()
            return 1.0 - surv_probs
        
        
        elif hasattr(model_obj, 'model') and hasattr(model_obj.model, 'predict_survival_function'):
            surv_funcs = model_obj.model.predict_survival_function(X_features)
            probs = []
            for fn in surv_funcs:
                try:
                    p = fn(t)
                except ValueError:
                    
                    
                    p = fn(fn.x[-1])
                probs.append(1.0 - p)
            return np.array(probs)
        else:
            raise AttributeError("无法识别的模型类型，不支持提取 15/20 年概率")
    # ============================================================


    n_folds = args.num_splits
    for i in range(n_folds):
        fold_ext = f"best_fold_{i}"
        print(f"\n--- Loading and Fitting Fold {i} ({fold_ext}) ---")
        transformer = load_preprocessor_from_file(args, fold_ext)
        X_train_p = process_data(X_train, args.cat_idx, transformer=transformer)
        X_val_p = process_data(X_val, args.cat_idx, transformer=transformer)
        X_asian_p = process_data(X_asian, args.cat_idx, transformer=transformer)
        X_other_p = process_data(X_other, args.cat_idx, transformer=transformer)
        model.load_model(fold_ext)
        
        
        model.fit_breslow(X_train_p, y_train)
        
        
        s_train, p_10_train = model.predict(X_train_p)
        s_val, p_10_val = model.predict(X_val_p)
        s_asian, p_10_asian = model.predict(X_asian_p)
        s_other, p_10_other = model.predict(X_other_p)
        
        
        
        p_15_train = safe_get_probs(model, X_train_p, s_train, time_point_15yr)
        p_20_train = safe_get_probs(model, X_train_p, s_train, time_point_20yr)
    
        p_15_val = safe_get_probs(model, X_val_p, s_val, time_point_15yr)
        p_20_val = safe_get_probs(model, X_val_p, s_val, time_point_20yr)
        
        p_15_asian = safe_get_probs(model, X_asian_p, s_asian, time_point_15yr)
        p_20_asian = safe_get_probs(model, X_asian_p, s_asian, time_point_20yr)
        
        p_15_other = safe_get_probs(model, X_other_p, s_other, time_point_15yr)
        p_20_other = safe_get_probs(model, X_other_p, s_other, time_point_20yr)

        
        ensemble_scores_train += s_train
        ensemble_scores_val += s_val
        ensemble_scores_asian += s_asian
        ensemble_scores_other += s_other

        ensemble_probs_10_train += p_10_train
        ensemble_probs_10_val += p_10_val
        ensemble_probs_10_asian += p_10_asian
        ensemble_probs_10_other += p_10_other

        
        ensemble_probs_15_train += p_15_train
        ensemble_probs_15_val += p_15_val
        ensemble_probs_15_asian += p_15_asian
        ensemble_probs_15_other += p_15_other

        ensemble_probs_20_train += p_20_train
        ensemble_probs_20_val += p_20_val
        ensemble_probs_20_asian += p_20_asian
        ensemble_probs_20_other += p_20_other

    
    avg_scores_train = ensemble_scores_train / n_folds
    avg_probs_10_train = ensemble_probs_10_train / n_folds
    avg_probs_15_train = ensemble_probs_15_train / n_folds
    avg_probs_20_train = ensemble_probs_20_train / n_folds

    avg_scores_val = ensemble_scores_val / n_folds
    avg_scores_asian = ensemble_scores_asian / n_folds
    avg_scores_other = ensemble_scores_other / n_folds

    avg_probs_10_val = ensemble_probs_10_val / n_folds
    avg_probs_10_asian = ensemble_probs_10_asian / n_folds
    avg_probs_10_other = ensemble_probs_10_other / n_folds

    avg_probs_15_val = ensemble_probs_15_val / n_folds
    avg_probs_15_asian = ensemble_probs_15_asian / n_folds
    avg_probs_15_other = ensemble_probs_15_other / n_folds

    avg_probs_20_val = ensemble_probs_20_val / n_folds
    avg_probs_20_asian = ensemble_probs_20_asian / n_folds
    avg_probs_20_other = ensemble_probs_20_other / n_folds

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("Saving predictions to disk...")
    
    save_dir = os.path.join(str(args.artifact_dir), "predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    clinical_feature_names = args.feature_names[-11:]
    
    def extract_clinical_features(X_data, col_names):
        if isinstance(X_data, pd.DataFrame):
            df_clin = X_data.iloc[:, -11:].copy()
            df_clin.columns = col_names
            return df_clin.reset_index(drop=True)
        elif isinstance(X_data, np.ndarray):
            df_clin = pd.DataFrame(X_data[:, -11:], columns=col_names)
            return df_clin
        else:
            raise ValueError(f"X 的数据类型不支持: {type(X_data)}")
    
    
    def create_result_df(y_true, scores, probs_10, probs_15, probs_20):  
        if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
            time_arr = y_true[:, 0]
            event_arr = y_true[:, 1]
        elif isinstance(y_true, np.ndarray) and y_true.dtype.names is not None:
            names = y_true.dtype.names
            event_name = names[0] if y_true.dtype[0].kind == 'b' else names[1]
            time_name = names[1] if y_true.dtype[0].kind == 'b' else names[0]
            event_arr = y_true[event_name]
            time_arr = y_true[time_name]
        elif isinstance(y_true, pd.DataFrame):
            time_arr = y_true.iloc[:, 0].values
            event_arr = y_true.iloc[:, 1].values
        else:
            raise ValueError(f"不支持的 y_true 格式")

        df = pd.DataFrame({
            'time': time_arr,
            'event': event_arr.astype(int),
            'risk_score': scores,
            'prob_10yr': probs_10,
            'prob_15yr': probs_15, 
            'prob_20yr': probs_20  
        })
        return df

    
    df_train_base = create_result_df(y_train, avg_scores_train, avg_probs_10_train, avg_probs_15_train, avg_probs_20_train)
    df_val_base = create_result_df(y_val, avg_scores_val, avg_probs_10_val, avg_probs_15_val, avg_probs_20_val)
    df_asian_base = create_result_df(y_asian, avg_scores_asian, avg_probs_10_asian, avg_probs_15_asian, avg_probs_20_asian)
    df_other_base = create_result_df(y_other, avg_scores_other, avg_probs_10_other, avg_probs_15_other, avg_probs_20_other)
    
    
    df_train_clin = extract_clinical_features(X_train, clinical_feature_names)
    df_val_clin = extract_clinical_features(X_val, clinical_feature_names)
    df_asian_clin = extract_clinical_features(X_asian, clinical_feature_names)
    df_other_clin = extract_clinical_features(X_other, clinical_feature_names)
    
    
    df_train = pd.concat([df_train_base, df_train_clin], axis=1)
    df_val = pd.concat([df_val_base, df_val_clin], axis=1)
    df_asian = pd.concat([df_asian_base, df_asian_clin], axis=1)
    df_other = pd.concat([df_other_base, df_other_clin], axis=1)
    
    # =========================================================
    
    # =========================================================
    model_folder = args.model_name
    outcome_folder = args.disease_type
    config_name = "classic" if args.data_type == "classic" else f"{args.data_type}_lassonet"
            
    final_save_dir = os.path.join(str(args.artifact_dir), "predictions", model_folder, outcome_folder)
    os.makedirs(final_save_dir, exist_ok=True)
    
    
    df_train.to_csv(os.path.join(final_save_dir, f"{config_name}_train.csv"), index=False)
    df_val.to_csv(os.path.join(final_save_dir, f"{config_name}_val.csv"), index=False)
    df_asian.to_csv(os.path.join(final_save_dir, f"{config_name}_asian.csv"), index=False)
    df_other.to_csv(os.path.join(final_save_dir, f"{config_name}_other.csv"), index=False)
    
    print(f"✅ 成功保存! 路径: {final_save_dir}")


def main_once(args):
    """
    超参数优化的主函数
    """
    print("Train model with given hyperparameters")
    
    
    X_eur, y_eur, X_asian, y_asian, X_other, y_other,Cat_index,feature_names = load_Survial_train_datas(
        args,
        protein_source = args.protein_source)
    args.feature_names = feature_names                                                                       
    args.cat_idx = Cat_index
    
    X_opt, X_val, y_opt, y_val = train_test_split(X_eur, y_eur, 
                                                  test_size=args.ratio,
                                                  random_state=args.seed,
                                                  stratify=y_eur[:, 1])
    set_category_dimensions(args, X_opt)
    
    
    model_name = str2model(args.model_name)

    
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    
    if args.data_type == 'classic':
        study_name = args.model_name + "_" + args.dataset + "_" + args.disease_type + "_" + args.data_type
    else:
        study_name = args.model_name + "_" + args.dataset + "_" + args.disease_type + "_" + args.data_type + "_" + args.protein_source
    
    storage_name = optuna_storage(args, study_name)
    args.study_name = study_name
    
    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    
    
    print("Best parameters:", study.best_trial.params)
    if "best_n_epochs" in study.best_trial.user_attrs:
        best_n_epochs = study.best_trial.user_attrs["best_n_epochs"]
    else:
        best_n_epochs = None
    print("best_n_epochs:", best_n_epochs)
    args.best_n_epochs = best_n_epochs
    
    model = model_name(study.best_trial.params, args)
    
    generate_and_save_predictions(model, 
                                  X_opt, y_opt, 
                                  X_val, y_val, 
                                  X_asian, y_asian, 
                                  X_other, y_other, args)

