import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

def ks_sample_test(
        sample1:np.ndarray, 
        sample2:np.ndarray):
    return ks_2samp(
        sample1, 
        sample2).statistic

def plot_side_by_side(
        scores1: pd.Series, 
        scores2: pd.Series, 
        title: str,
        ground_truth: pd.Series, 
        predictions: pd.Series, 
        pos_label: str = 'Sí',
        label1: str ='Recidivists', 
        label2: str ='Non recidivists'):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
    fig.suptitle('Histograms and ROC curve')
    ax1.set_title("Histograms")
    ax1.set_xlabel(title)
    ax1.set_ylabel("Probability")
    ax1.hist(scores1, histtype='step', label=label1, density=True, color='red')
    ax1.hist(scores2, histtype='step', label=label2, density=True, color='blue')
    ax1.legend(loc='upper right', fontsize='x-large')

    fpr, tpr, thresholds = roc_curve(ground_truth, predictions, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    ax2.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right", fontsize='x-large')
    plt.show()

def equal_error_rates(
        test_X:pd.DataFrame,
        y_test:pd.Series,
        y_pred:np.ndarray,
        category1:str,
        category2:str):
    pred_category1_recidivists = y_pred[(y_test=='No') & (test_X[category1]==1)]
    pred_category2_non_recidivists = y_pred[(y_test=='No') & (test_X[category2]==1)]
    return ks_sample_test(pred_category1_recidivists, pred_category2_non_recidivists)

def demographic_parity(
        test_X:pd.DataFrame,
        y_pred:np.ndarray,
        category1:str,
        category2:str):
    pred_category1_recidivists = y_pred[(test_X[category1]==1)]
    pred_category2_recidivists = y_pred[(test_X[category2]==1)]
    return ks_sample_test(pred_category1_recidivists, pred_category2_recidivists)

def equal_opportunity(
        test_X:pd.DataFrame,
        y_test:pd.Series,
        y_pred:np.ndarray,
        category1:str,
        category2:str):
    pred_category1_recidivists = y_pred[(y_test=='Sí') & (test_X[category1]==1)]
    pred_category2_recidivists = y_pred[(y_test=='Sí') & (test_X[category2]==1)]
    return ks_sample_test(pred_category1_recidivists, pred_category2_recidivists)