"""Generate out-of-fold predictions without reusing validation preprocessing."""

import os
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


from models.baseline_models import process_data
from models import str2model
from utils.load_data import load_Survial_train_datas, set_category_dimensions
from utils.parser import get_parser
from utils.paths import optuna_storage
from utils.io_utils import load_preprocessor_from_file

def generate_oof_scores(model, X_train, y_train, args):
    """
    核心 OOF 生成函数：只预测训练集，只保留 risk_score
    """
    print(f"Generating pure OOF risk scores for {args.model_name}...")
    n_train = len(y_train)
    oof_scores_train = np.zeros(n_train)

    
    if args.objective in ["classification", "binary", "Survival"]:
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
        y_for_split = y_train[:, 1].astype(int) if args.objective == "Survival" else y_train
    else:
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
        y_for_split = y_train

    for i, (train_index, val_index) in enumerate(kf.split(X_train, y_for_split)):
        fold_ext = f"best_fold_{i}"  
        print(f"--- Processing Fold {i} ({fold_ext}) ---")

        X_fold_train = X_train[train_index]
        y_fold_train = y_train[train_index]
        X_fold_val = X_train[val_index]

        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        transformer = load_preprocessor_from_file(args, fold_ext)
        X_fold_train_p = process_data(X_fold_train, args.cat_idx, transformer=transformer)
        X_fold_val_p = process_data(X_fold_val, args.cat_idx, transformer=transformer)

        
        model.load_model(fold_ext)
        
        
        model.fit_breslow(X_fold_train_p, y_fold_train)

        
        s_oof, _ = model.predict(X_fold_val_p)
        
        
        oof_scores_train[val_index] = s_oof

    return oof_scores_train


def generate_oof_for_config(args):
    """
    针对当前配置，加载数据并执行 OOF
    """
    
    X_eur, y_eur, _, _, _, _, Cat_index, feature_names = load_Survial_train_datas(
        args, protein_source=args.protein_source)
    args.feature_names = feature_names                                                                   
    args.cat_idx = Cat_index

    
    X_opt, _, y_opt, _ = train_test_split(X_eur, y_eur, 
                                          test_size=args.ratio,
                                          random_state=args.seed,
                                          stratify=y_eur[:, 1])
    set_category_dimensions(args, X_opt)

    
    if args.data_type == 'classic':
        study_name = f"{args.model_name}_{args.dataset}_{args.disease_type}_{args.data_type}"
    else:
        study_name = f"{args.model_name}_{args.dataset}_{args.disease_type}_{args.data_type}_{args.protein_source}"
    
    storage_name = optuna_storage(args, study_name)
    args.study_name = study_name

    print(f"\nLoading study parameters from: {study_name}.db")
    try:
        
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params = study.best_trial.params
        args.best_n_epochs = study.best_trial.user_attrs.get("best_n_epochs", None)
    except Exception as e:
        print(f"❌ 找不到数据库文件或加载失败: {e}。请确保在运行优化的同一目录下执行此脚本。")
        return

    
    model_class = str2model(args.model_name)
    model = model_class(best_params, args)

    
    oof_scores = generate_oof_scores(model, X_opt, y_opt, args)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    clinical_feature_names = args.feature_names[-11:]

    
    if isinstance(y_opt, np.ndarray) and y_opt.ndim == 2:
        time_arr = y_opt[:, 0]
        event_arr = y_opt[:, 1]
    elif isinstance(y_opt, np.ndarray) and y_opt.dtype.names is not None:
        names = y_opt.dtype.names
        event_name = names[0] if y_opt.dtype[0].kind == 'b' else names[1]
        time_name = names[1] if y_opt.dtype[0].kind == 'b' else names[0]
        event_arr = y_opt[event_name]
        time_arr = y_opt[time_name]
    else:
        time_arr = y_opt.iloc[:, 0].values
        event_arr = y_opt.iloc[:, 1].values

    
    df_base = pd.DataFrame({
        'time': time_arr,
        'event': event_arr.astype(int),
        'risk_score': oof_scores
    })

    
    if isinstance(X_opt, pd.DataFrame):
        df_clin = X_opt.iloc[:, -11:].copy()
        df_clin.columns = clinical_feature_names
        df_clin = df_clin.reset_index(drop=True)
    elif isinstance(X_opt, np.ndarray):
        df_clin = pd.DataFrame(X_opt[:, -11:], columns=clinical_feature_names)

    
    df_final = pd.concat([df_base, df_clin], axis=1)

    model_folder = args.model_name
    outcome_folder = args.disease_type
    config_name = "classic" if args.data_type == "classic" else f"{args.data_type}_lassonet"
    final_save_dir = os.path.join(str(args.artifact_dir), "oof_predictions", model_folder, outcome_folder)
    os.makedirs(final_save_dir, exist_ok=True)
    
    save_path = os.path.join(final_save_dir, f"{config_name}_oof_train.csv")
    df_final.to_csv(save_path, index=False)
    print(f"✅ 成功! 纯净版 OOF Score 已保存至: {save_path}\n")

