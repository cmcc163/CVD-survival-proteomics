"""Tune and fit one manuscript model for one outcome and predictor set."""

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
    save_preprocessor_to_file,
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

def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective in ["classification", "binary", "Survival"]:
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    
    
    y_for_split = y
    if args.objective == "Survival":
        y_for_split = y[:, 1].astype(int)

    epoch = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y_for_split)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_model = y_train
        y_test_model = y_test

        
        if args.model_name == 'LinearModel': 
            
            
            y_train_model = Surv.from_arrays(
                event=y_train[:, 1].astype(bool), 
                time=y_train[:, 0]
            )
            y_test_model = Surv.from_arrays(
                event=y_test[:, 1].astype(bool), 
                time=y_test[:, 0]
            )

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        
        
        if args.model_name in ['LinearModel', 'XGBoost']:
            
            X_train, my_transformer = process_data(X_train, args.cat_idx, method='onehot')
            X_test = process_data(X_test, args.cat_idx, transformer=my_transformer)
        else:
            X_train, my_transformer = process_data(X_train, args.cat_idx, method='deep')
            X_test = process_data(X_test, args.cat_idx, transformer=my_transformer)
            
        print(curr_model.params)
        loss_history, val_loss_history,best_epoch = curr_model.fit(X_train, y_train_model, 
                                                                   X_test, y_test_model,
                                                                   i)
        curr_model.save_model(f'temp_fold_{i}')
        save_preprocessor_to_file(my_transformer, args, f'temp_fold_{i}')
        
        if best_epoch is not None:
            epoch.append(best_epoch)
        train_timer.end()

        # Test model
        test_timer.start()
        risk_scores, event_probs = curr_model.predict(X_test)
        test_timer.end()

        # Compute scores on the output
        if args.objective == "Survival":
            sc.eval(y_test[:, 0], y_test[:, 1], risk_scores, event_probs)
        else:
            
            sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(sc.get_results())

    return sc, (train_timer.get_average_time(), test_timer.get_average_time()), epoch

def bootstrap_validation_average(model, 
                                 X_train, y_train, 
                                 X_val, y_val, 
                                 X_asian, y_asian, 
                                 X_other, y_other, 
                                 args, n_iterations=None):
    n_iterations = args.bootstrap_repeats if n_iterations is None else n_iterations
    print("Starting bootstrap validation with 5-Fold Ensemble...")
    
    
    sc_val = get_scorer(args)
    sc_asian = get_scorer(args)
    sc_other = get_scorer(args)
    
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    print("Loading each fold's fitted preprocessor...")

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    
    
    ensemble_probs_val = np.zeros(len(y_val))
    ensemble_probs_asian = np.zeros(len(y_asian))
    ensemble_probs_other = np.zeros(len(y_other))
    
    
    ensemble_scores_val = np.zeros(len(y_val))
    ensemble_scores_asian = np.zeros(len(y_asian))
    ensemble_scores_other = np.zeros(len(y_other))

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
        
        
        
        s_val, p_val = model.predict(X_val_p)
        s_asian, p_asian = model.predict(X_asian_p)
        s_other, p_other = model.predict(X_other_p)
        
        
        ensemble_scores_val += s_val
        ensemble_probs_val += p_val
        ensemble_scores_asian += s_asian
        ensemble_probs_asian += p_asian
        ensemble_scores_other += s_other
        ensemble_probs_other += p_other

    
    avg_scores_val = ensemble_scores_val / n_folds
    avg_probs_val = ensemble_probs_val / n_folds
    
    avg_scores_asian = ensemble_scores_asian / n_folds
    avg_probs_asian = ensemble_probs_asian / n_folds
    
    avg_scores_other = ensemble_scores_other / n_folds
    avg_probs_other = ensemble_probs_other / n_folds

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    def evaluate_ensemble_results(name, y, scores, probs, scorer):
        print(f"\n[Part {name}] Running {n_iterations} bootstrap on averaged predictions...")
        
        
        
        scorer.figure(y[:, 0], y[:, 1], probs, filename_extension=f"{name}_ensemble")
        
        
        y_time = y[:, 0]
        y_event = y[:, 1]
        n_samples = len(y)
        indices_all = np.arange(n_samples)

        for i in range(n_iterations):
            indices = resample(indices_all, replace=True, random_state=i)
            
            
            scorer.eval(
                y_time[indices], 
                y_event[indices], 
                scores[indices], 
                probs[indices]
            )
        
        
        results = scorer.get_results()
        print(f">>> {name} Ensemble Results: C-index Mean={results.get('C-index - mean', 'N/A')}")
        return results

    
    results_val = evaluate_ensemble_results("European", y_val, avg_scores_val, avg_probs_val, sc_val)
    results_asian = evaluate_ensemble_results("Asian", y_asian, avg_scores_asian, avg_probs_asian, sc_asian)
    results_other = evaluate_ensemble_results("Other", y_other, avg_scores_other, avg_probs_other, sc_other)

    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    final_results = {}
    
    
    for k, v in results_val.items():
        final_results[f"european_{k}"] = v
    for k, v in results_asian.items():
        final_results[f"asian_{k}"] = v
    for k, v in results_other.items():
        final_results[f"other_{k}"] = v

    
    
    if hasattr(args, 'best_n_epochs') and args.best_n_epochs is not None:
        model.params["best_n_epochs"] = args.best_n_epochs
    
    
    model.params["batch_size"] = getattr(args, 'batch_size', None)
    model.params["val_batch_size"] = getattr(args, 'val_batch_size', None)
    model.params["early_stopping_rounds"] = getattr(args, 'early_stopping_rounds', None)
    model.params["n_iterations_bootstrap"] = n_iterations
    model.params["ensemble_folds"] = n_folds

    
    print(f"\nSaving aggregated results to file...")
    save_results_to_file(
        args, 
        final_results,
        train_time=None,
        test_time=None,   
        best_params=model.params
    )

    print("Bootstrap validation and saving finished.")
    return final_results

def getmodel(model, 
             X, y, 
             X_val, y_val, 
             X_asian, y_asian, 
             X_other, y_other, 
             args, n_iterations=1000):
    print("Starting get models...")
    sc = get_scorer(args)
    
    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective in ["classification", "binary", "Survival"]:
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    
    
    y_for_split = y
    if args.objective == "Survival":
        y_for_split = y[:, 1].astype(int)

    epoch = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y_for_split)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_model = y_train
        y_test_model = y_test

        
        if args.model_name == 'LinearModel': 
            
            
            y_train_model = Surv.from_arrays(
                event=y_train[:, 1].astype(bool), 
                time=y_train[:, 0]
            )
            y_test_model = Surv.from_arrays(
                event=y_test[:, 1].astype(bool), 
                time=y_test[:, 0]
            )

        # Create a new unfitted version of the model
        curr_model = model.clone()

        
        if args.model_name in ['LinearModel', 'XGBoost']:
            
            X_train, my_transformer = process_data(X_train, args.cat_idx, method='onehot')
            X_test = process_data(X_test, args.cat_idx, transformer=my_transformer)
        else:
            X_train, my_transformer = process_data(X_train, args.cat_idx, method='deep')
            X_test = process_data(X_test, args.cat_idx, transformer=my_transformer)
            
        print(curr_model.params)
        loss_history, val_loss_history,best_epoch = curr_model.fit(X_train, y_train_model, 
                                                                   X_test, y_test_model,
                                                                   i)
        risk_scores, event_probs = curr_model.predict(X_test)
        
        # Compute scores on the output
        if args.objective == "Survival":
            sc.eval(y_test[:, 0], y_test[:, 1], risk_scores, event_probs)
        else:
            
            sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(sc.get_results())
        
        curr_model.save_model(f'best_fold_{i}')
        save_preprocessor_to_file(my_transformer, args, f'best_fold_{i}')
        
    print('finished!')

class Objective(object):
    """
    Optuna 的目标函数类，用于评估一组超参数的性能。
    Optuna 会多次调用这个类（即调用 __call__ 方法），每次传入不同的 trial（尝试）。
    """
    def __init__(self, args, model_name, X, y,study=None):
        
        self.model_name = model_name

        
        self.X = X
        self.y = y

        
        self.args = args
        
        
        self.best_score = None
        
        
        if study is not None:
            try:
                
                self.best_score = study.best_value
                print(f"🔄 Resuming study. Current historical best score: {self.best_score}")
            except ValueError:
                
                self.best_score = None
        
        
        if self.best_score is None:
            if args.direction == 'maximize':
                self.best_score = -float('inf')
            else:
                self.best_score = float('inf')

    def __call__(self, trial):
        """
        Optuna 每次优化的核心执行函数。
        
        参数:
        trial: Optuna 的 Trial 对象，用于建议（suggest）超参数。
        """
        
        
        
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        
        
        model = self.model_name(trial_params, self.args)

        
        
        
        sc, time, epoch_list = cross_validation(model, self.X, self.y, self.args)
        
        if epoch_list:
            avg_best_epoch = int(np.mean(epoch_list))
        else:
            avg_best_epoch = None
        
        trial.set_user_attr("best_n_epochs", avg_best_epoch)
        
        current_score = sc.get_objective_result()
        
        is_best = False
        if self.args.direction == 'maximize':
            if current_score >= self.best_score:
                is_best = True
        else: # minimize
            if current_score <= self.best_score:
                is_best = True
                
        
        if is_best:
            print(f"🌟 New Best Score: {current_score:.4f} (Previous: {self.best_score:.4f})")
            self.best_score = current_score

            
            
            
            dummy_path = get_output_path(self.args, directory="models", filename="dummy", extension="xx", file_type="placeholder")
            save_dir = os.path.dirname(dummy_path)
            
            if not os.path.exists(save_dir):
                print(f"Warning: Directory {save_dir} does not exist. Cannot save best model.")
                return current_score

            
            for i in range(self.args.num_splits):
                
                
                
                search_pattern = os.path.join(save_dir, f"*temp_fold_{i}.*")
                
                
                found_files = glob.glob(search_pattern)
                
                if not found_files:
                    
                    continue
                
                for f_path in found_files:
                    
                    
                    filename = os.path.basename(f_path)
                    
                    
                    
                    # m_temp_fold_0.pkl -> m_best_fold_0.pkl
                    new_filename = filename.replace(f"temp_fold_{i}", f"best_fold_{i}")
                    
                    dst_path = os.path.join(save_dir, new_filename)
                    
                    try:
                        
                        shutil.copy2(f_path, dst_path)
                        # print(f"  Saved: {new_filename}")
                    except Exception as e:
                        print(f"  Failed to copy best model {filename}: {e}")
                  
        
        trial_params["best_n_epochs"] = avg_best_epoch
        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        
        
        return current_score

def main(args):
    """
    超参数优化的主函数
    """
    print("Start hyperparameter optimization")
    
    X_eur, y_eur, X_asian, y_asian, X_other, y_other,Cat_index,feature_names = load_Survial_train_datas(
        args,
        protein_source = args.protein_source)
                                                                                                    
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

    
    
    
    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    
    
    
    
    study.optimize(Objective(args, model_name, X_opt, y_opt, study=study), 
                n_trials=args.n_trials,
                show_progress_bar=False,
                n_jobs=1)
    
    print("Best parameters:", study.best_trial.params)
    if "best_n_epochs" in study.best_trial.user_attrs:
        best_n_epochs = study.best_trial.user_attrs["best_n_epochs"]
    else:
        best_n_epochs = None
    args.best_n_epochs = best_n_epochs
    
    model = model_name(study.best_trial.params, args)
    
    
    bootstrap_validation_average(model, 
                                X_opt, y_opt, 
                                X_val, y_val, 
                                X_asian, y_asian, 
                                X_other, y_other, args)

def main_once(args):
    """
    超参数优化的主函数
    """
    print("Train model with given hyperparameters")
    
    
    X_eur, y_eur, X_asian, y_asian, X_other, y_other,Cat_index,feature_names = load_Survial_train_datas(args,
                                                                                                        protein_source = args.protein_source)
                                                                                                        
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
    
    bootstrap_validation_average(model, 
                                 X_opt, y_opt, 
                                 X_val, y_val, 
                                 X_asian, y_asian, 
                                 X_other, y_other, args)

