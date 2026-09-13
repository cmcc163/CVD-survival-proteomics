"""Compute TreeSHAP values from XGBoost surrogates of non-tree models."""

import os
import sys
import glob
import logging
import warnings
import optuna
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm  


from models.baseline_models import process_data
from models import str2model
from utils.load_data import load_Survial_train_datas, set_category_dimensions
from utils.parser import get_parser
from utils.paths import optuna_storage
from utils.io_utils import load_preprocessor_from_file


warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

def compute_unified_shap(args):
    """
    大一统 SHAP 计算核心：
    - 若模型为 XGBoost：直接使用 TreeExplainer 计算 5 折并取均值。
    - 若模型为其他 (MLP, TabNet, LinearModel等)：采用 5 折 OOF XGBoost Surrogate 代理模型策略。
    """
    print("\n" + "="*60)
    strategy = "Direct Ensemble" if args.model_name == 'XGBoost' else "Surrogate OOF"
    print(f"🚀 启动 [{args.model_name}] SHAP 分析 | 策略: {strategy}")
    print(f"Disease: {args.disease_type} | Data: {args.data_type} | Source: {args.protein_source}")
    print("="*60)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    data_source_str = "classic" if args.data_type == "classic" else f"{args.data_type}_lassonet"
    plot_dir = os.path.join(str(args.artifact_dir), "shap", args.disease_type, data_source_str, args.model_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    X_eur, y_eur, X_asian, y_asian, X_other, y_other, Cat_index, feature_names = load_Survial_train_datas(
        args, protein_source=args.protein_source
    )
    args.cat_idx = Cat_index
    
    X_opt_raw, X_holdout_raw, y_opt, y_holdout = train_test_split(
        X_eur, y_eur, 
        test_size=args.ratio,
        random_state=args.seed,
        stratify=y_eur[:, 1]
    )
    set_category_dimensions(args, X_opt_raw)
    
    X_holdout_raw_arr = X_holdout_raw.values if hasattr(X_holdout_raw, "values") else X_holdout_raw

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # The surrogate itself is trained only on the European development set.
    X_opt_xgb, my_transformer_x = process_data(X_opt_raw, args.cat_idx, method='onehot')
    X_val_xgb = process_data(X_holdout_raw, args.cat_idx, transformer=my_transformer_x)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    model_name_class = str2model(args.model_name)
    study_name = f"{args.model_name}_{args.dataset}_{args.disease_type}_{data_source_str}"
    storage_name = optuna_storage(args, study_name)
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params = study.best_trial.params
    except Exception as e:
        print(f"❌ 无法加载 Optuna 数据库 '{study_name}'，报错: {e}")
        return

    
    shap_values = None
    base_values = None
    interaction_values = None
    fidelity_r2 = None

    # =========================================================
    
    # =========================================================
    if args.model_name == 'XGBoost':
        print("  -> 正在使用 XGBoost 直接计算 5 折 SHAP (GPU 加速)...")
        n_folds = args.num_splits
        all_shap, all_base, all_interact = [], [], []
        
        for i in range(n_folds):
            model = model_name_class(best_params, args)
            model.load_model(f"best_fold_{i}")
            transformer = load_preprocessor_from_file(args, f"best_fold_{i}")
            X_val_fold_xgb = process_data(X_holdout_raw, args.cat_idx, transformer=transformer)
            
            xgb_core_model = model.model
            explainer = shap.TreeExplainer(xgb_core_model)
            shap_obj = explainer(X_val_fold_xgb)
            
            all_shap.append(shap_obj.values)
            all_base.append(shap_obj.base_values)
            all_interact.append(explainer.shap_interaction_values(X_val_fold_xgb))
            
        
        shap_values = np.mean(all_shap, axis=0)
        base_values = np.mean(all_base, axis=0)
        interaction_values = np.mean(all_interact, axis=0)

    # =========================================================
    
    # =========================================================
    else:
        print(f"  -> 正在提取 {args.model_name} 的 Teacher 预测目标 (OOF & Ensemble)...")
        oof_risk_scores = np.zeros(len(X_opt_raw))
        ensemble_risk_scores_holdout = np.zeros(len(X_holdout_raw))
        
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=args.seed)
        
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_opt_raw, y_opt[:, 1])):
            model = model_name_class(best_params, args)
            model.load_model(f"best_fold_{fold_idx}")
            transformer = load_preprocessor_from_file(args, f"best_fold_{fold_idx}")
            X_train_fold_teacher = process_data(X_opt_raw[train_index], args.cat_idx, transformer=transformer)
            X_val_fold_teacher = process_data(X_opt_raw[val_index], args.cat_idx, transformer=transformer)
            X_holdout_teacher = process_data(X_holdout_raw, args.cat_idx, transformer=transformer)
            model.fit_breslow(X_train_fold_teacher, y_opt[train_index])
            scores_oof, _ = model.predict(X_val_fold_teacher)
            oof_risk_scores[val_index] = scores_oof
            scores_holdout, _ = model.predict(X_holdout_teacher)
            ensemble_risk_scores_holdout += scores_holdout
            
        ensemble_risk_scores_holdout /= args.num_splits
        
        print(f"  -> 正在训练 XGBoost 高保真代理模型...")
        surrogate_model = xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.02, 
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', device='cuda' if args.use_gpu else 'cpu', random_state=args.seed, n_jobs=-1
        )
        
        
        surrogate_model.fit(X_opt_xgb, oof_risk_scores)
        
        
        surrogate_preds = surrogate_model.predict(X_val_xgb)
        fidelity_r2 = r2_score(ensemble_risk_scores_holdout, surrogate_preds)
        print(f"      🌟 代理模型保真度 (Fidelity R-squared): {fidelity_r2:.4f}")
        
        print("  -> 正在通过代理模型计算 TreeSHAP 及二阶交互作用...")
        explainer = shap.TreeExplainer(surrogate_model)
        shap_obj = explainer(X_val_xgb)
        shap_values = shap_obj.values
        base_values = shap_obj.base_values
        interaction_values = explainer.shap_interaction_values(X_val_xgb)

    # =========================================================
    
    # =========================================================
    print(f"\n  -> 正在生成统一的 Explanation 对象 ({X_val_xgb.shape[0]} 个样本全量)...")
    
    ensemble_explanation = shap.Explanation(
        values=shap_values, 
        base_values=base_values, 
        data=X_holdout_raw_arr,  
        feature_names=feature_names
    )

    if args.model_name == 'XGBoost':
        plot_title = f"{args.model_name} SHAP (Direct Ensemble)"
    else:
        plot_title = f"{args.model_name} SHAP (Surrogate Fidelity: R²={fidelity_r2:.3f})"

    print("  -> 正在生成可视化图表 (PNG 格式)...")
    
    
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(ensemble_explanation, max_display=20, show=False)
    plt.title(plot_title, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_beeswarm.png"), dpi=300)
    plt.close()
    
    
    plt.figure(figsize=(10, 6))
    shap.plots.bar(ensemble_explanation, max_display=20, show=False)
    plt.title(plot_title, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_bar.png"), dpi=300)
    plt.close()

    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(ensemble_explanation[0], max_display=15, show=False)
    plt.title(f"{plot_title} - Sample 0", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_waterfall_sample_0.png"), dpi=300)
    plt.close()

    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_3_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_3_features = [feature_names[i] for i in top_3_indices]
    
    for rank, feat_name in enumerate(top_3_features):
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(ensemble_explanation[:, feat_name], color=ensemble_explanation, show=False)
        plt.title(plot_title, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"shap_dependence_top{rank+1}_{feat_name}.png"), dpi=300)
        plt.close()

    
    mean_abs_interaction = np.mean(np.abs(interaction_values), axis=0)
    top_15_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_15_interact = mean_abs_interaction[np.ix_(top_15_idx, top_15_idx)]
    top_15_names = [feature_names[i] for i in top_15_idx]
    
    np.fill_diagonal(top_15_interact, 0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_15_interact, xticklabels=top_15_names, yticklabels=top_15_names, 
                cmap="viridis", annot=False)
    plt.title(f"{args.model_name} Mean Absolute SHAP Interaction (Top 15)\n" + 
              (f"Fidelity: R²={fidelity_r2:.3f}" if fidelity_r2 else "Direct Ensemble"), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_interaction_heatmap.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("  -> 正在将底层数据导出为 Excel...")
    excel_path = os.path.join(plot_dir, "SHAP_Raw_Data.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        df_importance = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)
        df_importance.to_excel(writer, sheet_name="Global_Importance", index=False)
        
        df_shap = pd.DataFrame(shap_values, columns=feature_names)
        df_shap.to_excel(writer, sheet_name="SHAP_Values_Matrix", index=False)
        
        df_feats = pd.DataFrame(X_holdout_raw_arr, columns=feature_names) 
        df_feats.to_excel(writer, sheet_name="Feature_Values_Original", index=False)
        
        base_val_0 = base_values[0] if isinstance(base_values, np.ndarray) else base_values
        df_sample0 = pd.DataFrame({
            "Feature": feature_names,
            "Original_Value": X_holdout_raw_arr[0], 
            "SHAP_Value": shap_values[0]
        })
        df_sample0.loc[-1] = ["[Base_Value_Reference]", np.nan, base_val_0]
        df_sample0.index = df_sample0.index + 1
        df_sample0.sort_index().to_excel(writer, sheet_name="Waterfall_Sample0", index=False)

        df_interact = pd.DataFrame(mean_abs_interaction, index=feature_names, columns=feature_names)
        df_interact.to_excel(writer, sheet_name="Mean_Abs_Interaction")

    print(f"🎉 全部完成！图表与数据均已保存至 {plot_dir}\n")

