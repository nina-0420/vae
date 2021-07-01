import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, cohen_kappa_score


def compute_metric(datanpGT, datanpPRED, target_names):

    n_class = len(target_names)
    argmaxPRED = np.argmax(datanpPRED, axis=1)
    F1_metric = np.zeros([n_class, 1])
    tn = np.zeros([n_class, 1])
    fp = np.zeros([n_class, 1])
    fn = np.zeros([n_class, 1])
    tp = np.zeros([n_class, 1])

    ROC_curve = {}
    mAUC = 0


    for i in range(n_class):

        tmp_label = datanpGT == i
        tmp_pred = argmaxPRED == i 

        F1_metric[i] = f1_score(tmp_label, tmp_pred)
        tn[i], fp[i], fn[i], tp[i] = confusion_matrix(tmp_label, tmp_pred).ravel()
        outAUROC = roc_auc_score(tmp_label, datanpPRED[:, i])

        mAUC = mAUC + outAUROC
        [roc_fpr, roc_tpr, roc_thresholds] = roc_curve(tmp_label, datanpPRED[:, i])

        ROC_curve.update({'ROC_fpr_'+str(i): roc_fpr,
                          'ROC_tpr_' + str(i): roc_tpr,
                          'ROC_T_' + str(i): roc_thresholds,
                          'AUC_' + str(i): outAUROC})

    mPrecision = sum(tp) / sum(tp + fp)   # precision or positive predictive value (PPV)
    mRecall = sum(tp) / sum(tp + fn)      # sensitivity, recall, hit rate, or true positive rate (TPR)
    mSpecificity = sum(tn) / sum(fp + tn) # specificity, selectivity or true negative rate (TNR)
    mAccuracy = sum(tp+tn) / sum(tn + fp + fn + tp)

    output = {
        'class_name': target_names,
        'F1': F1_metric,
        'AUC': mAUC / 3,
        'Accuracy': (tp + tn) / (tn + fp + fn + tp),
        'Sensitivity': tp / (tp + fn),
        'Precision': tp / (tp + fp),
        'Specificity': tn / (fp + tn),
        'ROC_curve': ROC_curve,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'mAccuracy': mAccuracy,
        'mPrecision': mPrecision,
        'mSensitivity': mRecall,
        'mSpecificity': mSpecificity,
        'mBalancedAcc': 0.5*(mRecall + mSpecificity),
        'mG_mean': np.sqrt(mRecall * mSpecificity),
        'mF1': 2*mPrecision * mRecall / (mPrecision + mRecall),
    }

    return output
