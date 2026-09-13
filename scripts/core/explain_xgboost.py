"""Compute direct fold-ensemble TreeSHAP values for XGBoost."""

import os
import sys
import glob
import logging
import optuna
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


from models.baseline_models import process_data
from models import str2model
from utils.load_data import load_Survial_train_datas, set_category_dimensions
from utils.parser import get_parser
from utils.paths import optuna_storage
from utils.io_utils import load_preprocessor_from_file

def compute_xgboost_ensemble_shap(args):
    """
    针对 XGBoost 计算 5 折交叉验证集成的 SHAP 值并绘图
    包含：原始值逆映射、交互作用计算、非线性依赖图、全量数据 Excel 导出
    """
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    data_source_str = "classic" if args.data_type == "classic" else f"{args.data_type}_lassonet"
    plot_dir = os.path.join(str(args.artifact_dir), "shap", args.disease_type, data_source_str)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n[{args.disease_type} - {data_source_str}] 开始 SHAP 计算流程...")
    print(f"结果将被保存至: {plot_dir}")
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    X_eur, y_eur, X_asian, y_asian, X_other, y_other, Cat_index, feature_names = load_Survial_train_datas(
        args, protein_source=args.protein_source
    )
    args.cat_idx = Cat_index
    
    
    X_opt, X_val, y_opt, y_val = train_test_split(
        X_eur, y_eur, 
        test_size=args.ratio,
        random_state=args.seed,
        stratify=y_eur[:, 1]
    )
    set_category_dimensions(args, X_opt)

    X_val_original = np.array(X_val)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    model_name = str2model(args.model_name)
    if args.data_type == 'classic':
        study_name = f"{args.model_name}_{args.dataset}_{args.disease_type}_{args.data_type}"
    else:
        study_name = f"{args.model_name}_{args.dataset}_{args.disease_type}_{args.data_type}_{args.protein_source}"
    storage_name = optuna_storage(args, study_name)
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params = study.best_trial.params
        print(f"  -> 成功加载 Optuna 最优参数: {study_name}")
    except Exception as e:
        print(f"❌ 错误: 无法加载 Optuna 数据库 '{study_name}'。报错信息: {e}")
        return

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    n_folds = args.num_splits
    all_fold_shap_values = []
    all_fold_base_values = []
    all_fold_interaction_values = []
    
    for i in range(n_folds):
        print(f"  -> 正在处理第 {i+1}/{n_folds} 折 (计算 Main & Interaction SHAP)...")
        model = model_name(best_params, args)
        model.load_model(f"best_fold_{i}")
        transformer = load_preprocessor_from_file(args, f"best_fold_{i}")
        X_val_p = process_data(X_val, args.cat_idx, transformer=transformer)
        if X_val_p.shape[1] != len(feature_names):
            raise ValueError(
                "The fold-specific encoded feature count does not match the published feature names."
            )
        xgb_core_model = model.model  
        
        
        explainer = shap.TreeExplainer(xgb_core_model)
        
        
        shap_obj = explainer(X_val_p)
        all_fold_shap_values.append(shap_obj.values)
        all_fold_base_values.append(shap_obj.base_values)
        
        
        interaction_vals = explainer.shap_interaction_values(X_val_p)
        all_fold_interaction_values.append(interaction_vals)
        
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("  -> 正在对多折结果求均值并构建 Explanation 对象...")
    mean_shap_values = np.mean(all_fold_shap_values, axis=0)
    mean_base_values = np.mean(all_fold_base_values, axis=0)
    mean_interaction_values = np.mean(all_fold_interaction_values, axis=0)
    
    
    ensemble_explanation = shap.Explanation(
        values=mean_shap_values, 
        base_values=mean_base_values, 
        data=X_val_original,  
        feature_names=feature_names
    )

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("  -> 正在生成并保存可视化图表...")
    
    
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(ensemble_explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_beeswarm.pdf"), dpi=300)
    plt.close()
    
    
    plt.figure(figsize=(10, 6))
    shap.plots.bar(ensemble_explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_bar.pdf"), dpi=300)
    plt.close()

    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(ensemble_explanation[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_waterfall_sample_0.pdf"), dpi=300)
    plt.close()

    
    mean_abs_shap = np.mean(np.abs(mean_shap_values), axis=0)
    top_3_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_3_features = [feature_names[i] for i in top_3_indices]
    
    for rank, feat_name in enumerate(top_3_features):
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(ensemble_explanation[:, feat_name], color=ensemble_explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"shap_dependence_top{rank+1}_{feat_name}.pdf"), dpi=300)
        plt.close()

    
    mean_abs_interaction = np.mean(np.abs(mean_interaction_values), axis=0)
    
    top_15_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_15_interact = mean_abs_interaction[np.ix_(top_15_idx, top_15_idx)]
    top_15_names = [feature_names[i] for i in top_15_idx]
    
    
    np.fill_diagonal(top_15_interact, 0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_15_interact, xticklabels=top_15_names, yticklabels=top_15_names, 
                cmap="viridis", annot=False)
    plt.title("Mean Absolute SHAP Interaction Values (Top 15 Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "shap_interaction_heatmap.pdf"), dpi=300)
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
        
        
        df_shap = pd.DataFrame(mean_shap_values, columns=feature_names)
        df_shap.to_excel(writer, sheet_name="SHAP_Values_Matrix", index=False)
        
        
        df_feats = pd.DataFrame(X_val_original, columns=feature_names)
        df_feats.to_excel(writer, sheet_name="Feature_Values_Original", index=False)
        
        
        base_val_0 = mean_base_values[0] if isinstance(mean_base_values, np.ndarray) else mean_base_values
        df_sample0 = pd.DataFrame({
            "Feature": feature_names,
            "Original_Value": X_val_original[0],
            "SHAP_Value": mean_shap_values[0]
        })
        
        df_sample0.loc[-1] = ["[Base_Value_Reference]", np.nan, base_val_0]
        df_sample0.index = df_sample0.index + 1
        df_sample0 = df_sample0.sort_index()
        df_sample0.to_excel(writer, sheet_name="Waterfall_Sample0", index=False)

        
        df_interact = pd.DataFrame(mean_abs_interaction, index=feature_names, columns=feature_names)
        df_interact.to_excel(writer, sheet_name="Mean_Abs_Interaction")

    print(f"🎉 全部完成！原始数据已导出至 {excel_path}\n")

