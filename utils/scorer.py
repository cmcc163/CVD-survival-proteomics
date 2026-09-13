from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix, average_precision_score, recall_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
import statsmodels.api as sm  
from utils.io_utils import get_output_path 
from sklearn.calibration import calibration_curve
import seaborn as sns
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter, KaplanMeierFitter


def get_scorer(args):
    if args.objective == "regression":
        return RegScorer()
    elif args.objective == "classification":
        return ClassScorer()
    elif args.objective == "binary":
        return BinScorer(args)
    elif args.objective == "Survival":
        return SurvScorer(args)
    else:
        raise NotImplementedError("No scorer for \"" + args.objective + "\" implemented")


class Scorer:

    """
        y_true: (n_samples,)
        y_prediction: (n_samples,) - predicted classes
        y_probabilities: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    """
    def eval(self, y_true, y_prediction, y_probabilities):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_objective_result(self):
        raise NotImplementedError("Has be implemented in the sub class")


class RegScorer(Scorer):

    def __init__(self):
        self.mses = []
        self.r2s = []

    # y_probabilities is None for Regression
    def eval(self, y_true, y_prediction, y_probabilities):
        mse = mean_squared_error(y_true, y_prediction)
        r2 = r2_score(y_true, y_prediction)

        self.mses.append(mse)
        self.r2s.append(r2)

        return {"MSE": mse, "R2": r2}

    def get_results(self):
        mse_mean = np.mean(self.mses)
        mse_std = np.std(self.mses)

        r2_mean = np.mean(self.r2s)
        r2_std = np.std(self.r2s)

        return {"MSE - mean": mse_mean,
                "MSE - std": mse_std,
                "R2 - mean": r2_mean,
                "R2 - std": r2_std}

    def get_objective_result(self):
        return np.mean(self.mses)


class ClassScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []
        self.sensitivities = []
        self.specificities = []
        self.pr_aucs = []

    def eval(self, y_true, y_prediction, y_probabilities):
        logloss = log_loss(y_true, y_probabilities)
        # auc = roc_auc_score(y_true, y_probabilities, multi_class='ovr')
        auc = roc_auc_score(y_true, y_probabilities, multi_class='ovo', average="macro")

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="weighted")  # use here macro or weighted?

        sensitivity = recall_score(y_true, y_prediction, average='macro')

        n_classes = y_probabilities.shape[1]
        cm = confusion_matrix(y_true, y_prediction, labels=np.arange(n_classes))
        specificities = []
        for c in range(n_classes):
            tn = np.sum(cm) - np.sum(cm[c, :]) - np.sum(cm[:, c]) + cm[c, c]
            fp = np.sum(cm[:, c]) - cm[c, c]
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.mean(specificities)

        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        if n_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        pr_auc = average_precision_score(y_true_bin, y_probabilities, average="macro")

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)
        self.sensitivities.append(sensitivity)
        self.specificities.append(specificity)
        self.pr_aucs.append(pr_auc)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1,
                "Sensitivity": sensitivity, "Specificity": specificity, "PR-AUC": pr_auc}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        sensitivity_mean = np.mean(self.sensitivities)
        sensitivity_std = np.std(self.sensitivities)

        specificity_mean = np.mean(self.specificities)
        specificity_std = np.std(self.specificities)

        pr_auc_mean = np.mean(self.pr_aucs)
        pr_auc_std = np.std(self.pr_aucs)

        results = {
            "Log Loss - mean": logloss_mean,
            "Log Loss - std": logloss_std,
            "AUC - mean": auc_mean,
            "AUC - std": auc_std,
            "Accuracy - mean": acc_mean,
            "Accuracy - std": acc_std,
            "F1 score - mean": f1_mean,
            "F1 score - std": f1_std,
            "Sensitivity - mean": sensitivity_mean,
            "Sensitivity - std": sensitivity_std,
            "Specificity - mean": specificity_mean,
            "Specificity - std": specificity_std,
            "PR-AUC - mean": pr_auc_mean,
            "PR-AUC - std": pr_auc_std
        }

        
        if len(self.aucs) >= 100:
            metrics_map = {
                "AUC": self.aucs,
                "Accuracy": self.accs,
                "F1 score": self.f1s,
                "Sensitivity": self.sensitivities,
                "Specificity": self.specificities,
                "PR-AUC": self.pr_aucs
            }

            for name, values in metrics_map.items():
                lower = np.percentile(values, 2.5)
                upper = np.percentile(values, 97.5)
                
                results[f"{name} - 95% CI"] = (lower, upper)

        return results

    def get_objective_result(self):
        return np.mean(self.loglosses)



def set_pub_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,           
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
    })


COLORS = {
    'blue': '#3C5488',    
    'red': '#E64B35',     
    'gray': '#7F7F7F',    
    'light_gray': '#D3D3D3', 
    'fill_pos': '#4DBBD5', 
    'fill_neg': '#E64B35', 
}


class SurvScorer:

    def __init__(self, args):
        self.args = args
        self.c_indices = []
        self.aucs_10yr = []
        self.cal_slopes = []
        self.oe_ratios = []
        self.net_benefits = []
        self.sensitivities = []
        self.specificities = []
        self.ppvs = []
        self.npvs = []
        
        
        self.time_horizon = 3652.0

    def eval(self, y_time, y_event, y_risk_score, y_prob_10yr, threshold=0.10):
        
        y_time = np.array(y_time).flatten()
        y_event = np.array(y_event).flatten().astype(bool)
        y_risk_score = np.array(y_risk_score).flatten()
        y_prob_10yr = np.array(y_prob_10yr).flatten()
        
        # --- 1. C-index ---
        try:
            c_index = concordance_index_censored(y_event, y_time, y_risk_score)[0]
        except Exception:
            c_index = np.nan

        # --- 2. 10yr AUC ---
        y_struct = np.array([(e, t) for e, t in zip(y_event, y_time)], 
                            dtype=[('e', bool), ('t', float)])
        try:
            if self.time_horizon < np.min(y_time) or self.time_horizon > np.max(y_time):
                auc_10 = np.nan
            else:
                auc_res = cumulative_dynamic_auc(y_struct, y_struct, y_risk_score, [self.time_horizon])
                auc_10 = auc_res[0][0]
        except Exception:
            auc_10 = np.nan

        # --- 3. Calibration ---
        # Slope
        try:
            if np.std(y_risk_score) > 1e-6:
                df_cal = pd.DataFrame({'T': y_time, 'E': y_event, 'pred': y_risk_score})
                cph = CoxPHFitter()
                cph.fit(df_cal, duration_col='T', event_col='E')
                cal_slope = cph.params_['pred']
            else:
                cal_slope = np.nan
        except:
            cal_slope = np.nan

        # O:E Ratio
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(y_time, y_event)
            
            if self.time_horizon > np.max(y_time):
                 obs_risk_10 = 1 - kmf.survival_function_.iloc[-1, 0]
            else:
                 obs_risk_10 = 1 - kmf.predict(self.time_horizon)
                 
            pred_risk_mean = np.mean(y_prob_10yr)
            oe_ratio = obs_risk_10 / pred_risk_mean if pred_risk_mean > 1e-6 else 0
        except:
            oe_ratio = np.nan
            obs_risk_10 = 0 # Fallback

        # --- 4. Net Benefit & Metrics (KM weighted) ---
        mask_pos = y_prob_10yr >= threshold
        mask_neg = ~mask_pos
        n = len(y_time)
        n_pos = np.sum(mask_pos)
        
        try:
            
            if n_pos > 10: 
                kmf_pos = KaplanMeierFitter()
                kmf_pos.fit(y_time[mask_pos], y_event[mask_pos])
                risk_in_pos = 1 - kmf_pos.predict(self.time_horizon)
            else:
                
                risk_in_pos = np.mean(y_event[mask_pos]) if n_pos > 0 else 0

            
            if np.sum(mask_neg) > 10:
                kmf_neg = KaplanMeierFitter()
                kmf_neg.fit(y_time[mask_neg], y_event[mask_neg])
                risk_in_neg = 1 - kmf_neg.predict(self.time_horizon)
            else:
                risk_in_neg = np.mean(y_event[mask_neg]) if np.sum(mask_neg) > 0 else 0

            
            prob_pos = n_pos / n
            tp_rate = risk_in_pos * prob_pos
            fp_rate = (1 - risk_in_pos) * prob_pos
            
            w = threshold / (1 - threshold)
            net_benefit = tp_rate - fp_rate * w
            
            
            ppv = risk_in_pos
            npv = 1 - risk_in_neg
            
            # Sensitivity = TP / Total_Events
            total_event_rate = obs_risk_10
            sens = tp_rate / total_event_rate if total_event_rate > 0 else 0
            
            # Specificity = TN / Total_Non_Events
            tn_rate = (1 - total_event_rate) - fp_rate
            spec = tn_rate / (1 - total_event_rate) if (1 - total_event_rate) > 0 else 0
            
        except Exception as e:
            print(f"Metrics calc error: {e}")
            net_benefit, sens, spec, ppv, npv = 0, 0, 0, 0, 0

        
        self.c_indices.append(c_index)
        self.aucs_10yr.append(auc_10)
        self.cal_slopes.append(cal_slope)
        self.oe_ratios.append(oe_ratio)
        self.net_benefits.append(net_benefit)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.ppvs.append(ppv)
        self.npvs.append(npv)

        return {
            "C-index": c_index,
            "AUC (10yr)": auc_10,
            "Net Benefit": net_benefit
        }

    def get_results(self):
        """计算所有指标的均值、标准差及95% CI"""
        results = {}
        
        metrics_map = {
            "C-index": self.c_indices,
            "AUC (10yr)": self.aucs_10yr,
            "Calibration Slope": self.cal_slopes,
            "O:E Ratio": self.oe_ratios,
            "Net Benefit": self.net_benefits,
            "Sensitivity": self.sensitivities,
            "Specificity": self.specificities,
            "PPV": self.ppvs,
            "NPV": self.npvs
        }

        for name, values in metrics_map.items():
            clean_values = [v for v in values if not np.isnan(v)]
            if not clean_values:
                continue
                
            results[f"{name} - mean"] = np.mean(clean_values)
            results[f"{name} - std"] = np.std(clean_values)
            
            if len(clean_values) >= 10:
                lower = round(np.percentile(clean_values, 2.5), 5)
                upper = round(np.percentile(clean_values, 97.5), 5)
                results[f"{name} - 95% CI"] = (lower, upper)

        return results

    def get_objective_result(self):
        
        return np.mean(self.c_indices)

    def figure(self, y_time, y_event, y_prob_10yr, filename_extension="", directory="figures", plot_type='all'):
        """
        分别绘制并保存 10年 节点的评估图 (单图单文件，正方形)
        """
        set_pub_style()
        
        y_time = np.array(y_time)
        y_event = np.array(y_event, dtype=bool)
        y_prob = np.array(y_prob_10yr)
        
        y_struct = np.array([(e, t) for e, t in zip(y_event, y_time)], 
                            dtype=[('e', bool), ('t', float)])

        
        if plot_type == 'all':
            target_plots = ['roc', 'calibration', 'risk_dist', 'dca']
        else:
            target_plots = [plot_type]

        
        for p_type in target_plots:
            
            
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

            
            def beautify_ax(ax, grid=True):
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)
                ax.set_box_aspect(1) 
                if grid:
                    ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['light_gray'], zorder=0)
                else:
                    ax.grid(False)

            # ==========================
            #      A. ROC Curve
            # ==========================
            if p_type == 'roc':
                try:
                    auc_res = cumulative_dynamic_auc(y_struct, y_struct, y_prob, [self.time_horizon])
                    auc_val = auc_res[0][0]
                    
                    y_viz_true = np.zeros_like(y_event, dtype=int)
                    y_viz_true[(y_time <= self.time_horizon) & (y_event == 1)] = 1
                    mask_valid = ~((y_time < self.time_horizon) & (y_event == 0))
                    
                    if np.sum(mask_valid) > 10:
                        fpr, tpr, _ = roc_curve(y_viz_true[mask_valid], y_prob[mask_valid])
                        ax.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}', lw=2.5, color=COLORS['blue'])
                        ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['blue'])
                    
                    ax.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--', lw=1.5)
                    ax.set_title('ROC Curve (10 Years)', fontweight='bold')
                    ax.set_xlabel('1 - Specificity')
                    ax.set_ylabel('Sensitivity')
                    ax.set_xlim([-0.02, 1.02])
                    ax.set_ylim([-0.02, 1.02])
                    ax.legend(loc='lower right', frameon=False)
                except Exception as e:
                    print(f"ROC Error: {e}")
                
                beautify_ax(ax)

            # ==========================
            #      B. Calibration
            # ==========================
            elif p_type == 'calibration':
                ax.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--', label='Ideal', zorder=1)
                try:
                    n_bins = 10
                    df_viz = pd.DataFrame({'prob': y_prob, 'time': y_time, 'event': y_event})
                    try:
                        df_viz['bin'] = pd.qcut(df_viz['prob'], n_bins, duplicates='drop')
                    except:
                        df_viz['bin'] = pd.cut(df_viz['prob'], bins=n_bins)

                    obs_probs = []
                    pred_probs = []
                    ci_errs = []
                    
                    kmf_cal = KaplanMeierFitter()
                    
                    for _, group in df_viz.groupby('bin', observed=True):
                        if len(group) == 0: continue
                        pred_mean = group['prob'].mean()
                        kmf_cal.fit(group['time'], group['event'])
                        if self.time_horizon > kmf_cal.timeline[-1]:
                            obs_rate = 1 - kmf_cal.survival_function_.iloc[-1, 0]
                        else:
                            obs_rate = 1 - kmf_cal.predict(self.time_horizon)
                        
                        se = np.sqrt(obs_rate * (1 - obs_rate) / len(group))
                        obs_probs.append(obs_rate)
                        pred_probs.append(pred_mean)
                        ci_errs.append(1.96 * se)
                    
                    ax.errorbar(pred_probs, obs_probs, yerr=ci_errs, fmt='o', color='black',
                                markersize=5, capsize=3, elinewidth=1.5,
                                label='Observed (KM)', mfc='white', mew=1.5, zorder=2)
                    
                    
                    max_val = max(max(pred_probs), max(obs_probs))
                    limit = min(1.0, max(0.2, max_val + 0.1)) 
                    ax.set_xlim(0, limit)
                    ax.set_ylim(0, limit)
                except Exception as e:
                    print(f"Calibration Error: {e}")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                ax.set_title('Calibration (10 Years)', fontweight='bold')
                ax.set_xlabel('Predicted Risk')
                ax.set_ylabel('Observed Incidence')
                ax.legend(loc='upper left', frameon=False)
                beautify_ax(ax)

            # ==========================
            #    C. Risk Distribution
            # ==========================
            elif p_type == 'risk_dist':
                labels = []
                valid_indices = []
                for i, (t, e) in enumerate(zip(y_time, y_event)):
                    if e == 1 and t <= self.time_horizon:
                        labels.append('Event <10yr')
                        valid_indices.append(i)
                    elif t > self.time_horizon:
                        labels.append('Event-free >10yr')
                        valid_indices.append(i)
                
                if len(labels) < 10:
                    labels = ['Event' if e else 'Censored' for e in y_event]
                    valid_indices = range(len(y_event))

                df_dist = pd.DataFrame({'prob': y_prob[valid_indices], 'label': labels})
                palette = {labels[0]: COLORS['fill_pos'], labels[1]: COLORS['fill_neg']}
                if 'Event <10yr' in labels: palette['Event <10yr'] = COLORS['red']
                if 'Event-free >10yr' in labels: palette['Event-free >10yr'] = COLORS['blue']

                sns.violinplot(data=df_dist, x='label', y='prob', ax=ax, hue='label', legend=False,
                               palette=palette, inner=None, linewidth=0, alpha=0.4, cut=0)
                sns.boxplot(data=df_dist, x='label', y='prob', ax=ax, width=0.15,
                            boxprops={'facecolor':'white', 'edgecolor': 'black', 'alpha':0.8},
                            showfliers=False, zorder=3)
                
                ax.set_title('Risk Distribution', fontweight='bold')
                ax.set_ylabel('Predicted Risk')
                ax.set_xlabel('')
                beautify_ax(ax, grid=False)
                ax.grid(axis='y', linestyle='--', alpha=0.3)

            # ==========================
            #         D. DCA
            # ==========================
            elif p_type == 'dca':
                thresholds = np.linspace(0.01, 0.60, 60)
                net_benefit_model = []
                net_benefit_all = []
                
                kmf_all = KaplanMeierFitter()
                kmf_all.fit(y_time, y_event)
                prevalence = 1 - kmf_all.predict(self.time_horizon)
                n_total = len(y_time)

                for t in thresholds:
                    w = t / (1 - t)
                    mask_high = y_prob >= t
                    n_high = np.sum(mask_high)
                    
                    if n_high > 10: 
                        kmf_sub = KaplanMeierFitter()
                        kmf_sub.fit(y_time[mask_high], y_event[mask_high])
                        risk_high = 1 - kmf_sub.predict(self.time_horizon)
                        tp_rate = risk_high * (n_high / n_total)
                        fp_rate = (1 - risk_high) * (n_high / n_total)
                        nb = tp_rate - fp_rate * w
                    else:
                        nb = 0 
                    
                    net_benefit_model.append(nb)
                    nb_all = prevalence - (1 - prevalence) * w
                    net_benefit_all.append(nb_all)

                ax.plot(thresholds, net_benefit_model, label='Model', color=COLORS['red'], lw=2.5)
                ax.plot(thresholds, net_benefit_all, label='Treat All', color=COLORS['gray'], linestyle=':', lw=1.5)
                ax.axhline(y=0, color='black', lw=1, label='Treat None')
                
                ax.set_title('Decision Curve (10 Years)', fontweight='bold')
                ax.set_xlabel('Threshold Probability')
                ax.set_ylabel('Net Benefit')
                y_max = max(0.1, prevalence + 0.05)
                ax.set_ylim(-0.01, y_max) 
                ax.set_xlim(0, 0.5)
                ax.legend(frameon=False, loc='upper right')
                beautify_ax(ax)

            # ==========================
            #        Save Logic
            # ==========================
            
            
            current_ext = f"{filename_extension}_{p_type}"
            
            save_path = get_output_path(self.args, directory=directory, filename="surv_perf", 
                                        extension=current_ext, file_type="png")
            
            plt.savefig(save_path, dpi=300)
            plt.close()


class BinScorer:

    def __init__(self,args):
        self.args = args
        
        self.aucs = []
        self.cal_slopes = []
        self.oe_ratios = []
        self.net_benefits = []
        self.sensitivities = []
        self.specificities = []
        self.ppvs = []
        self.npvs = []

    def eval(self, y_true, y_prediction, y_probabilities, threshold=0.075):
        """
        :param y_true: 真实标签
        :param y_prediction: (已忽略) 不使用硬分类标签，强制基于 threshold 重新计算
        :param y_probabilities: 预测概率 (N, 2) 或 (N,)
        :param threshold: 临床决策阈值 (ASCVD 推荐 0.075 或 0.1)
        """
        
        y_prob = y_probabilities[:, 1] if y_probabilities.ndim > 1 else y_probabilities
        y_true = np.array(y_true)

        
        auc = roc_auc_score(y_true, y_prob)

        
        # 2.1 O:E Ratio
        obs_rate = np.mean(y_true)
        pred_rate = np.mean(y_prob)
        oe_ratio = obs_rate / pred_rate if pred_rate > 0 else 0

        
        try:
            epsilon = 1e-9
            y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
            log_odds = logit(y_prob_clipped).reshape(-1, 1)
            
            
            lr = LogisticRegression(penalty=None, solver='lbfgs')
            lr.fit(log_odds, y_true)
            cal_slope = lr.coef_[0][0]
        except:
            cal_slope = np.nan  

        
        
        y_pred_clin = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_clin).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        
        # Net Benefit = (TP/N) - (FP/N) * (pt / (1-pt))
        n = len(y_true)
        w = threshold / (1 - threshold)
        net_benefit = (tp / n) - (fp / n) * w

        
        self.aucs.append(auc)
        self.cal_slopes.append(cal_slope)
        self.oe_ratios.append(oe_ratio)
        self.net_benefits.append(net_benefit)
        self.sensitivities.append(sens)
        self.specificities.append(spec)
        self.ppvs.append(ppv)
        self.npvs.append(npv)

        return {
            "AUROC": auc,
            "Calibration Slope": cal_slope,
            "O:E Ratio": oe_ratio,
            "Net Benefit": net_benefit,
            "Sensitivity": sens,
            "Specificity": spec,
            "PPV": ppv,
            "NPV": npv
        }

    def get_results(self):
        """计算所有指标的均值、标准差及95% CI (若样本足够)"""
        results = {}
        
        metrics_map = {
            "AUROC": self.aucs,
            "Calibration Slope": self.cal_slopes,
            "O:E Ratio": self.oe_ratios,
            "Net Benefit": self.net_benefits,
            "Sensitivity": self.sensitivities,
            "Specificity": self.specificities,
            "PPV": self.ppvs,
            "NPV": self.npvs
        }

        for name, values in metrics_map.items():
            clean_values = [v for v in values if not np.isnan(v)]
            if not clean_values:
                continue
                
            results[f"{name} - mean"] = np.mean(clean_values)
            results[f"{name} - std"] = np.std(clean_values)
            
            
            if len(clean_values) >= 10:
                lower = round(np.percentile(clean_values, 2.5), 5)
                upper = round(np.percentile(clean_values, 97.5), 5)
                results[f"{name} - 95% CI"] = (lower, upper)

        return results

    def get_objective_result(self):
        
        return np.mean(self.aucs)

    def figure(self, y_true, y_probabilities, filename_extension="", directory="figures", plot_type='all'):
        
        y_prob = y_probabilities[:, 1] if y_probabilities.ndim > 1 else y_probabilities
        y_true = np.array(y_true)

        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Liberation Serif', 'serif'],
            'font.sans-serif': ['Arial', 'Liberation Sans', 'sans-serif'],
            'mathtext.fontset': 'stix',
            'axes.linewidth': 1.2,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.size': 10
        })

        title_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}
        label_font = {'family': 'sans-serif', 'weight': 'bold', 'size': 11}

        
        if plot_type == 'all':
            
            fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=300) 
            axes = axes.flatten()
            
            fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.95, top=0.92, bottom=0.08)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
            axes = [ax]

        def style_ax(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            ax.set_box_aspect(1) 

        # --- A. Discrimination ---
        if plot_type in ['roc', 'all']:
            ax = axes[0]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}', lw=2, color='#00468B')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.6)
            ax.set_title('A. Discrimination', fontdict=title_font, loc='left', pad=10)
            ax.set_xlabel('1 - Specificity', fontdict=label_font)
            ax.set_ylabel('Sensitivity', fontdict=label_font)
            ax.legend(loc='lower right', frameon=False)
            style_ax(ax)

        # --- B. Calibration ---
        if plot_type in ['calibration', 'all']:
            ax = axes[1]
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.6, label='Ideal')
            
            
            
            n_bins = 10
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
            
            
            bin_size = len(y_true) / n_bins
            se = np.sqrt(prob_true * (1 - prob_true) / bin_size)
            
            ax.errorbar(prob_pred, prob_true, yerr=1.96*se, fmt='o', color='black', 
                        markersize=5, capsize=3, elinewidth=1, label='Observed (95% CI)', 
                        linestyle='none', mfc='white', mec='black', zorder=3)
            
            # Logistic Calibration (Platt Scaling)
            try:
                epsilon = 1e-9
                p_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
                x_logit = logit(p_clipped).reshape(-1, 1)
                lr_cal = LogisticRegression(penalty=None)
                lr_cal.fit(x_logit, y_true)
                x_range = np.linspace(0.001, 0.999, 100)
                y_calibrated = lr_cal.predict_proba(logit(x_range).reshape(-1, 1))[:, 1]
                ax.plot(x_range, y_calibrated, color='#ED0000', lw=2, label='Logistic Calibration')
            except: pass
            
            ax.set_title('B. Calibration', fontdict=title_font, loc='left', pad=10)
            ax.set_xlabel('Predicted Probability', fontdict=label_font)
            ax.set_ylabel('Observed Proportion', fontdict=label_font)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(loc='upper left', frameon=False, prop={'size': 9})
            style_ax(ax)

        # --- C. Risk Distribution ---
        if plot_type in ['risk_dist', 'all']:
            ax = axes[2]
            df_viz = pd.DataFrame({'prob': y_prob, 'label': ['ASCVD' if y==1 else 'Non-ASCVD' for y in y_true]})
            palette = {'Non-ASCVD': '#42B540', 'ASCVD': '#ED0000'}
            
            sns.violinplot(data=df_viz, x='label', y='prob', ax=ax, hue='label', legend=False,
                        palette=palette, inner=None, linewidth=1, cut=0)
            sns.boxplot(data=df_viz, x='label', y='prob', ax=ax, width=0.15, 
                        boxprops={'facecolor':'white', 'edgecolor':'black', 'alpha':0.6},
                        showfliers=False, zorder=3)
            
            ax.set_title('C. Risk Distribution', fontdict=title_font, loc='left', pad=10)
            ax.set_ylabel('Predicted Probability', fontdict=label_font)
            ax.set_xlabel('')
            ax.set_xticks([0, 1]) 
            ax.set_xticklabels(['Non-ASCVD', 'ASCVD'], fontdict={'family':'sans-serif', 'weight':'bold'})
            style_ax(ax)

        # --- D. Decision Curve Analysis (DCA) ---
        if plot_type in ['dca', 'all']:
            ax = axes[3]
            thresholds = np.linspace(0.01, 0.40, 50) 
            net_benefit_model = []
            net_benefit_all = []
            n = len(y_true)
            prevalence = np.mean(y_true)
            
            for t in thresholds:
                tp = np.sum((y_prob >= t) & (y_true == 1))
                fp = np.sum((y_prob >= t) & (y_true == 0))
                w = t / (1 - t)
                net_benefit_model.append((tp / n) - (fp / n) * w)
                net_benefit_all.append(prevalence - (1 - prevalence) * w)

            ax.plot(thresholds, net_benefit_model, label='Model', color='#ED0000', lw=2)
            ax.plot(thresholds, net_benefit_all, label='Treat All', color='gray', linestyle=':', lw=1.5)
            ax.axhline(y=0, color='black', lw=1, label='Treat None')
            
            ax.set_title('D. Decision Curve Analysis', fontdict=title_font, loc='left', pad=10)
            ax.set_xlabel('Threshold Probability', fontdict=label_font)
            ax.set_ylabel('Net Benefit', fontdict=label_font)
            ax.set_xlim(0, 0.4)
            ax.set_ylim(-0.01, prevalence + 0.05)
            ax.legend(frameon=False)
            style_ax(ax)

        
        save_path = get_output_path(self.args, directory=directory, filename="perf", 
                                    extension=filename_extension, file_type="png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight') 
        plt.close()
