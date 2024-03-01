import pandas as pd
import numpy as np
import math

from utils.distances import Distances
from sklearn.metrics import f1_score, accuracy_score


def getTPRandFPRbyThreshold(validation_scores):
    unique_scores = np.arange(0, 1, 0.01)
    total_positive = len(validation_scores[validation_scores["class"] == 1])
    total_negative = len(validation_scores[validation_scores["class"] == 0])

    arrayOfTPRandFPRByTr = None

    for threshold in unique_scores:
        fp = len(
            validation_scores[
                (validation_scores["score"] > threshold)
                & (validation_scores["class"] == 0)
            ]
        )
        tp = len(
            validation_scores[
                (validation_scores["score"] > threshold)
                & (validation_scores["class"] == 1)
            ]
        )
        tpr = round(tp / total_positive, 2)
        fpr = round(fp / total_negative, 2)

        aux = pd.DataFrame(data=[[round(threshold, 2), fpr, tpr]], columns=[
                           "threshold", "fpr", "tpr"])

        if arrayOfTPRandFPRByTr is None:
            arrayOfTPRandFPRByTr = aux
        else:
            arrayOfTPRandFPRByTr = pd.concat([arrayOfTPRandFPRByTr, aux])

    arrayOfTPRandFPRByTr.reset_index(inplace=True, drop=True)

    return arrayOfTPRandFPRByTr


def find_tprfpr_by_threshold(tprfpr, threshold):
    tprfpr_threshold = {}
    instance = tprfpr.query(f"threshold == {threshold}")
    tprfpr_threshold["fpr"] = instance["fpr"].iloc[0].astype(float)
    tprfpr_threshold["tpr"] = instance["tpr"].iloc[0].astype(float)
    return tprfpr_threshold


def get_hist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins) + 1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks, 1.1)

    re = np.repeat(1 / (len(breaks) - 1), (len(breaks) - 1))
    for i in range(1, len(breaks)):
        re[i - 1] = (
            re[i - 1]
            + len(np.where((scores >= breaks[i - 1]) & (scores < breaks[i]))[0])
        ) / (len(scores) + 1)

    return re


def DyS_distance(sc_1, sc_2, measure):
    """This function applies a selected distance metric"""

    dist = Distances(sc_1, sc_2)

    if measure == "topsoe":
        return dist.topsoe()
    if measure == "probsymm":
        return dist.probsymm()
    if measure == "hellinger":
        return dist.hellinger()
    return 100


def ternary_search(left, right, f, eps=1e-4):
    """This function applies Ternary search"""

    while True:
        if abs(left - right) < eps:
            return (left + right) / 2

        leftThird = left + (right - left) / 3
        rightThird = right - (right - left) / 3

        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird


def find_best_threshold(scores, pred_prop, threshold=0.5, tolerance=0.01):
    low = 0.0
    high = 1.0
    max_iterations = math.ceil(math.log2(len(scores))) * 2
    thresholds = {}
    for _ in range(max_iterations):
        positive_proportion = sum(1 for score in scores if score > threshold) / len(
            scores
        )
        error = round(abs(positive_proportion - pred_prop), 3)
        if error not in thresholds:
            thresholds[error] = threshold
        if error < tolerance:
            return threshold
        if positive_proportion > pred_prop:
            low = threshold
            threshold = (threshold + high) / 2
        else:
            high = threshold
            threshold = (threshold + low) / 2
    threshold = thresholds[min(thresholds.keys())]
    return threshold


def calculate_accuracy(test_sample, test_label, pred_prop):
    scores_list = test_sample.to_list()
    sorted_list = sorted(scores_list)

    best_threshold = find_best_threshold(sorted_list, pred_prop)

    y_true = test_label.to_list()
    y_pred = [1 if score > best_threshold else 0 for score in scores_list]
    acc = accuracy_score(y_true, y_pred)
    return round(acc, 2), round(best_threshold, 2)


def calculate_fmeasure(test_sample, test_label, best_threshold):
    scores_list = test_sample.to_list()
    y_true = test_label.to_list()
    y_pred = [1 if score > best_threshold else 0 for score in scores_list]
    f_measure = f1_score(y_true, y_pred, zero_division=np.nan)
    return round(f_measure, 2)
