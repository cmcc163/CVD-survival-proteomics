"""Calculate PREVENT-equation risks for the clinical reference analysis."""

import logging
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


from utils.load_data import load_Survial_train_datas
from utils.scorer import get_scorer
from utils.io_utils import save_results_to_file
from utils.parser import get_parser, get_given_parameters_parser

# =============================================================================

# =============================================================================
class PREVENTModel_Total_CVD:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def predict(self, X_input):
        
        if not isinstance(X_input, pd.DataFrame):
            df = pd.DataFrame(X_input, columns=self.feature_names)
        else:
            df = X_input.copy()

        
        age_trans = (df['age'] - 55) / 10
        non_hdl_trans = df['non_hdl_cholesterol'] - 3.5
        hdl_trans = (df['hdl_cholesterol'] - 1.3) / 0.3
        sbp_trans_1 = (np.minimum(df['average_SBP'], 110) - 110) / 20
        sbp_trans_2 = (np.maximum(df['average_SBP'], 110) - 130) / 20
        egfr_trans_1 = (np.minimum(df['eGFR'], 60) - 60) / -15
        egfr_trans_2 = (np.maximum(df['eGFR'], 60) - 90) / -15

        diabetes = df['Diabetes_baseline']
        smoker = df['ever_smoked']
        htn_meds = df['hypertension_treatment']
        statin = df['Cholesterol_treatment']

        log_odds = np.zeros(len(df))

        
        fem_idx = df['sex'] == 0
        if fem_idx.any():
            lo_fem = -3.307728
            lo_fem += 0.7939329 * age_trans[fem_idx]
            lo_fem += 0.0305239 * non_hdl_trans[fem_idx]
            lo_fem += -0.1606857 * hdl_trans[fem_idx]
            lo_fem += -0.2394003 * sbp_trans_1[fem_idx]
            lo_fem += 0.360078 * sbp_trans_2[fem_idx]
            lo_fem += 0.8667604 * diabetes[fem_idx]
            lo_fem += 0.5360739 * smoker[fem_idx]
            lo_fem += 0.6045917 * egfr_trans_1[fem_idx]
            lo_fem += 0.0433769 * egfr_trans_2[fem_idx]
            lo_fem += 0.3151672 * htn_meds[fem_idx]
            lo_fem += -0.1477655 * statin[fem_idx]
            
            lo_fem += -0.0663612 * (htn_meds[fem_idx] * sbp_trans_2[fem_idx])
            lo_fem += 0.1197879 * (statin[fem_idx] * non_hdl_trans[fem_idx])
            lo_fem += -0.0819715 * (age_trans[fem_idx] * non_hdl_trans[fem_idx])
            lo_fem += 0.0306769 * (age_trans[fem_idx] * hdl_trans[fem_idx])
            lo_fem += -0.0946348 * (age_trans[fem_idx] * sbp_trans_2[fem_idx])
            lo_fem += -0.27057 * (age_trans[fem_idx] * diabetes[fem_idx])
            lo_fem += -0.078715 * (age_trans[fem_idx] * smoker[fem_idx])
            lo_fem += -0.1637806 * (age_trans[fem_idx] * egfr_trans_1[fem_idx])
            log_odds[fem_idx] = lo_fem

        
        male_idx = df['sex'] == 1
        if male_idx.any():
            lo_male = -3.031168
            lo_male += 0.7688528 * age_trans[male_idx]
            lo_male += 0.0736174 * non_hdl_trans[male_idx]
            lo_male += -0.0954431 * hdl_trans[male_idx]
            lo_male += -0.4347345 * sbp_trans_1[male_idx]
            lo_male += 0.3362658 * sbp_trans_2[male_idx]
            lo_male += 0.7692857 * diabetes[male_idx]
            lo_male += 0.4386871 * smoker[male_idx]
            lo_male += 0.5378979 * egfr_trans_1[male_idx]
            lo_male += 0.0164827 * egfr_trans_2[male_idx]
            lo_male += 0.288879 * htn_meds[male_idx]
            lo_male += -0.1337349 * statin[male_idx]
            
            lo_male += -0.0475924 * (htn_meds[male_idx] * sbp_trans_2[male_idx])
            lo_male += 0.150273 * (statin[male_idx] * non_hdl_trans[male_idx])
            lo_male += -0.0517874 * (age_trans[male_idx] * non_hdl_trans[male_idx])
            lo_male += 0.0191169 * (age_trans[male_idx] * hdl_trans[male_idx])
            lo_male += -0.1049477 * (age_trans[male_idx] * sbp_trans_2[male_idx])
            lo_male += -0.2251948 * (age_trans[male_idx] * diabetes[male_idx])
            lo_male += -0.0895067 * (age_trans[male_idx] * smoker[male_idx])
            lo_male += -0.1543702 * (age_trans[male_idx] * egfr_trans_1[male_idx])
            log_odds[male_idx] = lo_male

        risk_probs = np.exp(log_odds) / (1 + np.exp(log_odds))
        return log_odds, risk_probs

    def save_predictions(self, y, name):
        pass

class PREVENTModel_HF:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def predict(self, X_input):
        if not isinstance(X_input, pd.DataFrame):
            df = pd.DataFrame(X_input, columns=self.feature_names)
        else:
            df = X_input.copy()

        age_trans = (df['age'] - 55) / 10
        sbp_trans_1 = (np.minimum(df['average_SBP'], 110) - 110) / 20
        sbp_trans_2 = (np.maximum(df['average_SBP'], 110) - 130) / 20
        egfr_trans_1 = (np.minimum(df['eGFR'], 60) - 60) / -15
        egfr_trans_2 = (np.maximum(df['eGFR'], 60) - 90) / -15
        bmi_trans_1 = (np.minimum(df['BMI'], 30) - 25) / 5
        bmi_trans_2 = (np.maximum(df['BMI'], 30) - 30) / 5

        diabetes = df['Diabetes_baseline']
        smoker = df['ever_smoked']
        htn_meds = df['hypertension_treatment']

        log_odds = np.zeros(len(df))

        # Women
        fem_idx = df['sex'] == 0
        if fem_idx.any():
            lo_fem = -4.310409
            lo_fem += 0.8998235 * age_trans[fem_idx]
            lo_fem += -0.4559771 * sbp_trans_1[fem_idx]
            lo_fem += 0.3576505 * sbp_trans_2[fem_idx]
            lo_fem += 1.038346 * diabetes[fem_idx]
            lo_fem += 0.583916 * smoker[fem_idx]
            lo_fem += -0.0072294 * bmi_trans_1[fem_idx]
            lo_fem += 0.2997706 * bmi_trans_2[fem_idx]
            lo_fem += 0.7451638 * egfr_trans_1[fem_idx]
            lo_fem += 0.0557087 * egfr_trans_2[fem_idx]
            lo_fem += 0.3534442 * htn_meds[fem_idx]
            
            lo_fem += -0.0981511 * (htn_meds[fem_idx] * sbp_trans_2[fem_idx])
            lo_fem += -0.0946663 * (age_trans[fem_idx] * sbp_trans_2[fem_idx])
            lo_fem += -0.3581041 * (age_trans[fem_idx] * diabetes[fem_idx])
            lo_fem += -0.1159453 * (age_trans[fem_idx] * smoker[fem_idx])
            lo_fem += -0.003878 * (age_trans[fem_idx] * bmi_trans_2[fem_idx])
            lo_fem += -0.1884289 * (age_trans[fem_idx] * egfr_trans_1[fem_idx])
            log_odds[fem_idx] = lo_fem

        # Men
        male_idx = df['sex'] == 1
        if male_idx.any():
            lo_male = -3.946391
            lo_male += 0.8972642 * age_trans[male_idx]
            lo_male += -0.6811466 * sbp_trans_1[male_idx]
            lo_male += 0.3634461 * sbp_trans_2[male_idx]
            lo_male += 0.923776 * diabetes[male_idx]
            lo_male += 0.5023736 * smoker[male_idx]
            lo_male += -0.0485841 * bmi_trans_1[male_idx]
            lo_male += 0.3726929 * bmi_trans_2[male_idx]
            lo_male += 0.6926917 * egfr_trans_1[male_idx]
            lo_male += 0.0251827 * egfr_trans_2[male_idx]
            lo_male += 0.2980922 * htn_meds[male_idx]
            
            lo_male += -0.0497731 * (htn_meds[male_idx] * sbp_trans_2[male_idx])
            lo_male += -0.1289201 * (age_trans[male_idx] * sbp_trans_2[male_idx])
            lo_male += -0.3040924 * (age_trans[male_idx] * diabetes[male_idx])
            lo_male += -0.1401688 * (age_trans[male_idx] * smoker[male_idx])
            lo_male += 0.0068126 * (age_trans[male_idx] * bmi_trans_2[male_idx])
            lo_male += -0.1797778 * (age_trans[male_idx] * egfr_trans_1[male_idx])
            log_odds[male_idx] = lo_male

        risk_probs = np.exp(log_odds) / (1 + np.exp(log_odds))
        return log_odds, risk_probs

    def save_predictions(self, y, name):
        pass

class PREVENTModel_ASCVD:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def predict(self, X_input):
        if not isinstance(X_input, pd.DataFrame):
            df = pd.DataFrame(X_input, columns=self.feature_names)
        else:
            df = X_input.copy()

        age_trans = (df['age'] - 55) / 10
        non_hdl_trans = df['non_hdl_cholesterol'] - 3.5
        hdl_trans = (df['hdl_cholesterol'] - 1.3) / 0.3
        sbp_trans_1 = (np.minimum(df['average_SBP'], 110) - 110) / 20
        sbp_trans_2 = (np.maximum(df['average_SBP'], 110) - 130) / 20
        egfr_trans_1 = (np.minimum(df['eGFR'], 60) - 60) / -15
        egfr_trans_2 = (np.maximum(df['eGFR'], 60) - 90) / -15

        diabetes = df['Diabetes_baseline']
        smoker = df['ever_smoked']
        htn_meds = df['hypertension_treatment']
        statin = df['Cholesterol_treatment']

        log_odds = np.zeros(len(df))

        # Women
        fem_idx = df['sex'] == 0
        if fem_idx.any():
            lo_fem = -3.819975
            lo_fem += 0.719883 * age_trans[fem_idx]
            lo_fem += 0.1176967 * non_hdl_trans[fem_idx]
            lo_fem += -0.151185 * hdl_trans[fem_idx]
            lo_fem += -0.0835358 * sbp_trans_1[fem_idx]
            lo_fem += 0.3592852 * sbp_trans_2[fem_idx]
            lo_fem += 0.8348585 * diabetes[fem_idx]
            lo_fem += 0.4831078 * smoker[fem_idx]
            lo_fem += 0.4864619 * egfr_trans_1[fem_idx]
            lo_fem += 0.0397779 * egfr_trans_2[fem_idx]
            lo_fem += 0.2265309 * htn_meds[fem_idx]
            lo_fem += -0.0592374 * statin[fem_idx]
            
            lo_fem += -0.0395762 * (htn_meds[fem_idx] * sbp_trans_2[fem_idx])
            lo_fem += 0.0844423 * (statin[fem_idx] * non_hdl_trans[fem_idx])
            lo_fem += -0.0567839 * (age_trans[fem_idx] * non_hdl_trans[fem_idx])
            lo_fem += 0.0325692 * (age_trans[fem_idx] * hdl_trans[fem_idx])
            lo_fem += -0.1035985 * (age_trans[fem_idx] * sbp_trans_2[fem_idx])
            lo_fem += -0.2417542 * (age_trans[fem_idx] * diabetes[fem_idx])
            lo_fem += -0.0791142 * (age_trans[fem_idx] * smoker[fem_idx])
            lo_fem += -0.1671492 * (age_trans[fem_idx] * egfr_trans_1[fem_idx])
            log_odds[fem_idx] = lo_fem

        # Men
        male_idx = df['sex'] == 1
        if male_idx.any():
            lo_male = -3.500655
            lo_male += 0.7099847 * age_trans[male_idx]
            lo_male += 0.1658663 * non_hdl_trans[male_idx]
            lo_male += -0.1144285 * hdl_trans[male_idx]
            lo_male += -0.2837212 * sbp_trans_1[male_idx]
            lo_male += 0.3239977 * sbp_trans_2[male_idx]
            lo_male += 0.7189597 * diabetes[male_idx]
            lo_male += 0.3956973 * smoker[male_idx]
            lo_male += 0.3690075 * egfr_trans_1[male_idx]
            lo_male += 0.0203619 * egfr_trans_2[male_idx]
            lo_male += 0.2036522 * htn_meds[male_idx]
            lo_male += -0.0865581 * statin[male_idx]
            
            lo_male += -0.0322916 * (htn_meds[male_idx] * sbp_trans_2[male_idx])
            lo_male += 0.114563 * (statin[male_idx] * non_hdl_trans[male_idx])
            lo_male += -0.0300005 * (age_trans[male_idx] * non_hdl_trans[male_idx])
            lo_male += 0.0232747 * (age_trans[male_idx] * hdl_trans[male_idx])
            lo_male += -0.0927024 * (age_trans[male_idx] * sbp_trans_2[male_idx])
            lo_male += -0.2018525 * (age_trans[male_idx] * diabetes[male_idx])
            lo_male += -0.0970527 * (age_trans[male_idx] * smoker[male_idx])
            lo_male += -0.1217081 * (age_trans[male_idx] * egfr_trans_1[male_idx])
            log_odds[male_idx] = lo_male

        risk_probs = np.exp(log_odds) / (1 + np.exp(log_odds))
        return log_odds, risk_probs

    def save_predictions(self, y, name):
        pass


# =============================================================================

# =============================================================================
def save_prevent_predictions(args, model, datasets):
    """
    保存 PREVENT 模型的预测结果到 CSV
    datasets: 字典 {'val': (X,y), 'asian': (X,y), 'other': (X,y)}
    """
    print("=" * 30)
    print("Saving PREVENT predictions to CSV...")
    print("=" * 30)

    
    
    
    save_dir = os.path.join(str(args.artifact_dir), "predictions", "PREVENT", args.disease_type)
    os.makedirs(save_dir, exist_ok=True)
    
    config_name = "classic" 

    for split_name, (X_data, y_data) in datasets.items():
        print(f"Processing {split_name} data...")
        
        
        
        
        risk_scores, risk_probs = model.predict(X_data)
        
        
        if isinstance(y_data, np.ndarray):
            time_arr = y_data[:, 0]
            event_arr = y_data[:, 1]
        elif isinstance(y_data, pd.DataFrame):
            time_arr = y_data.iloc[:, 0].values
            event_arr = y_data.iloc[:, 1].values
        else:
            print(f"Warning: y_data type {type(y_data)} not supported for {split_name}")
            continue

        df_base = pd.DataFrame({
            'time': time_arr,
            'event': event_arr.astype(int),
            'risk_score': risk_scores,
            'prob_10yr': risk_probs
        })
        
        
        
        if isinstance(X_data, pd.DataFrame):
            df_features = X_data.reset_index(drop=True)
        else:
            df_features = pd.DataFrame(X_data, columns=args.feature_names)
            
        
        df_final = pd.concat([df_base, df_features], axis=1)
        
        
        filename = f"{config_name}_{split_name}.csv"
        file_path = os.path.join(save_dir, filename)
        df_final.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

    print("All predictions saved successfully.")

# =============================================================================

# =============================================================================
def main(args):
    # -----------------------------------------------------
    
    # -----------------------------------------------------
    args.protein_source = 'classic'
    X_eur, y_eur, X_asian, y_asian, X_other, y_other, Cat_index, feature_names = load_Survial_train_datas(
        args = args,
        protein_source = args.protein_source)
    
    args.cat_idx = Cat_index
    args.feature_names = feature_names 
    
    
    X_opt, X_val, y_opt, y_val = train_test_split(X_eur, y_eur, 
                                                  test_size=args.ratio,
                                                  random_state=args.seed,
                                                  stratify=y_eur[:, 1])
    
    print(f"Data Loaded. Validation size: {X_val.shape}, Asian size: {X_asian.shape}, Other size: {X_other.shape}")

    # -----------------------------------------------------
    
    # -----------------------------------------------------
    if args.disease_type == "Total_CVD":
        curr_model = PREVENTModel_Total_CVD(feature_names=feature_names)
    elif args.disease_type == "HF":
        curr_model = PREVENTModel_HF(feature_names=feature_names)
    elif args.disease_type == "ASCVD": 
        curr_model = PREVENTModel_ASCVD(feature_names=feature_names)

    # -----------------------------------------------------
    
    # -----------------------------------------------------
    
    datasets_to_save = {
        'val': (X_val, y_val),
        'asian': (X_asian, y_asian),
        'other': (X_other, y_other)
    }
    
    save_prevent_predictions(args, curr_model, datasets_to_save)

    print("Done.")

