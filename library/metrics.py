"""
Functions used to compute various classification metrics
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, f1_score, confusion_matrix

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_metrics_from_labels(true_label, predict_label):
    computed_metrics = dict(
        accuracy    = accuracy_score(true_label, predict_label),
        cohen_kappa = cohen_kappa_score(true_label, predict_label),
        sensitivity = recall_score(true_label, predict_label, average = 'weighted'),
        f1          = f1_score(true_label, predict_label, average = 'weighted'),
        confusion_matrix = compute_confusion_matrix(true_label, predict_label)
    )
    
    if len(set(true_label)) > 2 : 
        specificity = compute_specificity_multiclass(true_label, predict_label)
    else : 
        specificity = compute_specificity_binary(true_label, predict_label)
    computed_metrics["specificity"] = specificity
        
    
    return computed_metrics

def compute_specificity_multiclass(true_label, predict_label, weight_sum = True):
    """
    Compute the average specificity
    """

    binary_specificity_list = []
    weight_list = []

    for label in set(true_label):
        # Create binary label for the specific class
        tmp_true_label = (true_label == label).int()
        tmp_predict_label = (predict_label == label).int()
        
        # Compute specificity
        binary_specificity_list.append(compute_specificity_binary(tmp_true_label, tmp_predict_label))
        
        # (OPTIONAL) Count the number of example for the specific class
        if weight_sum: weight_list.append(int(tmp_true_label.sum()))
        else: weight_list.append(1)

    return np.average(binary_specificity_list, weights = weight_list)


def compute_specificity_binary(true_label, predict_label):
    # Get confusion matrix
    cm = confusion_matrix(true_label, predict_label)
    
    # Get True Negative and False positive
    TN = cm[1, 1]
    FP = cm[1, 0]
    
    # Compute specificity
    specificity = TN / (TN + FP)

    return specificity

def compute_confusion_matrix(true_label, predict_label):
    # Create the confusion matrix
    confusion_matrix = np.zeros((len(np.unique(true_label)), len(np.unique(true_label))))
    
    # Iterate through labels
    for i in range(len(true_label)):
        # Get the true and predicts labels
        # Notes that the labels are saved as number from 0 to 3 so can be used as index
        tmp_true = true_label[i]
        tmp_predict = predict_label[i]
        
        confusion_matrix[tmp_true, tmp_predict] += 1
    
    # Normalize between 0 and 1
    confusion_matrix /= len(true_label)
    
    return confusion_matrix
