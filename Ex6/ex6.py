"""
Author: Hao Zheng
Matr.Nr.: K01608113
Exercise 6
"""

'''function should return tuple with 4 values:
1. CM  a nested list [[TP, FN], [FP, TN]]
2. the F1-score as Python float object (set to 0 in case of a division by 0 error)
3. the accuracy as Python float object
4. the balanced accuracy as Python float object

function input:
1. logits: torch.tensor of shape (n samples), dtpye float
2. activation_function
3. threshold: torch.tensor any dtype
4. targets: torch.tensor of shape (n_samples) dtype: torch.bool


Your function is only allowed to use PyTorch to compute the scores. The computation of
scores must be performed using torch.float64 datatype.

'''

import torch


def ex6(logits: torch.Tensor, activation_function, threshold: torch.Tensor, targets: torch.Tensor):
    if torch.is_floating_point(logits) is False:
        raise TypeError("Logits datatype is not float")
    if isinstance(threshold, torch.Tensor) is False:
        raise TypeError("Threshold is not a torch.Tensor")
    if targets.dtype is not torch.bool:
        raise TypeError("Datatype of targets is not torch.bool ")

    n_samples = len(logits)
    if logits.shape != (n_samples,) or targets.shape != (n_samples,):
        raise ValueError(f"Shape of logits or targets is not ({n_samples}, )")
    if len(logits) != len(targets):
        raise ValueError(f'Length of logits and target does not match! {len(logits)} != {len(targets)}')
    if (True in targets and False in targets) is False:
        raise ValueError(
            "targets does not contain at least one entry with value False and at least one entry with value True.")
    # apply activation function
    nn_output = activation_function(logits)
    predictions = nn_output >= threshold
    # calculate evaluation metrics:
    pos = len(targets[targets == True])
    neg = len(targets[targets == False])
    tp = torch.sum(torch.logical_and(predictions == True, targets == True))
    tn = torch.sum(torch.logical_and(predictions == False, targets == False))
    fp = torch.sum(torch.logical_and(predictions == True, targets == False))
    fn = torch.sum(torch.logical_and(predictions == False, targets == True))

    tpr = tp / pos
    tnr = tn / neg

    if (2 * tp + fp + fn) == 0:
        f1_score = float(0)
    else:
        f1_score = float((2 * tp) / (2 * tp + fp + fn))

    acc = float((tp + tn) / n_samples)
    b_acc = float((tpr + tnr) / 2)

    return [[tp, fn], [fp, tn]], f1_score, acc, b_acc
