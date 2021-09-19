"""
functions for Measure the error for classification:
    accuracy,confusion_matrix,recall,precision,false_positive_rate,F_score

"""
import numpy as np


def accuracy(y, predict):
    y, predict = _prepare_data(y), _prepare_data(predict)
    return np.mean(predict == y)


def null_accuracy(y):
    y = _prepare_data(y)
    k = np.unique(y)
    return np.mean(np.array(y)[:, None] == k, axis=0)


def confusion_matrix(y, predict, count_mode=False):
    """
    compute the confusion matrix for classification
    rows->true labels
    columns->ML_ prediction labels

    :param
        y: true label for examples
        predict: predict label for examples

    :return: confusion matrix

    :complexity: O(m^2) where m in number of examples
    """
    y, predict = _prepare_data(y), _prepare_data(predict)
    m, k = y.shape[0], np.unique(y).shape[0]
    y, predict = np.array(y[:], dtype=np.uint8).reshape((-1,)), np.array(predict[:], dtype=np.uint8).reshape((-1,))
    M = np.zeros((k, k), dtype=np.float64)
    np.add.at(M, (y, predict), 1)
    if count_mode:
        return M.astype(np.int)
    else:
        return M / m


def recall(y, predict):
    """
    compute the recall from the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :Formula: recall = true_positive/(true_positive+false_negative)


    :return: recall vector for every class [recall(class A),...,recall(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    M = confusion_matrix(y, predict)
    return np.diag(M) / (np.sum(M, axis=1))


def precision(y, predict):
    """
    compute the precision from the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :Formula: precision = true_positive/(true_positive+false_positive)

    :return: precision vector for every class [precision(class A),...,precision(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    M = confusion_matrix(y, predict)
    return np.diag(M) / (np.sum(M, axis=0))


def false_positive_rate(y, predict):
    """
    compute the false positive rate from the confusion matrix for classification

    :param
        y: true label for examples
        predict: predict label for examples

    :Formula: FPR = false_positive/(false_positive+true_negative)

    :return: precision vector for every class [FPR(class A),...,FPR(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    M = confusion_matrix(y, predict)
    FP = np.sum(M, axis=0) - np.diag(M)
    TN = np.sum(np.diag(M)) - np.diag(M)
    return FP / (FP + TN)


def F_score(y, predict, beta=1):
    """
    compute the F_score from the confusion matrix,precision and recall for classification

    :param
        y: true label for examples
        predict: predict label for examples
        beta: if |beta| < 1 : more weight for the recall
              if |beta| = 1 : same weight for precision and recall (= harmonic mean)
              if |beta| > 1 : more weight for the precision

            :note: if beta==1: F1 score



    :Formula: F_score = ((1+beta^2) * recall * precision) / (beta ** 2 * precision + recall)
        (harmonic measure of recall and precision)

    :return: precision vector for every class [F_score(class A),...,F_score(class K)]

    :efficiency: O(m^2) where m in number of examples
    """
    recall_, precision_ = recall(y, predict), precision(y, predict)
    return ((1 + beta ** 2) * recall_ * precision_) / (beta ** 2 * precision_ + recall_)


def roc(y, p, k, thresholds_num=100):
    """
    calculate the connection between the recall ond false positive rate (=False alarms)
    according to the chosen threshold
    for good ML_ the recall is close to 1 even where the FTP is small

    :param
        y: true label for examples
        predict: predict label for examples
        k: number of classes
        thresholds_num: number of thresholds

    :return:
        false positive rate, true positive rate, thresholds
    """
    # fpr, tpr = np.empty((thresholds_num,)), np.empty((thresholds_num,))
    # thresholds = np.linspace(start=0, stop=1, num=thresholds_num, endpoint=True)[::-1] ** 22
    #
    # predict = predict[:, k_pos]
    # for i, x in enumerate(thresholds):
    #     p = predict > x
    #     fpr[i], tpr[i] = false_positive_rate(y, p)[k_pos], recall(y, p)[k_pos]
    #
    # idx = np.argsort(fpr).reshape((-1,))
    # return fpr[idx], tpr[idx], thresholds[idx]
    fpr, tpr = np.empty((thresholds_num, k)), np.empty((thresholds_num, k))
    thresholds = np.linspace(start=0, stop=1, num=thresholds_num, endpoint=True)[::-1] ** 22

    for i, x in enumerate(thresholds):
        p = p > x
        fpr[i, :], tpr[i, :] = false_positive_rate(y, p), recall(y, p)

    idx = fpr.argsort(axis=0), np.arange(fpr.shape[1])
    return fpr[idx], tpr[idx], thresholds, idx


def auc(y, p, k, k_pos, thresholds_num=100):
    """
    calculate the area beneath the graph of roc_auc if area > 0.5 ML_ is good

    :param
        y: true label for examples
        predict: predict label for examples
        k_pos: class chosen for test
        thresholds_num: number of thresholds

    :return:
        area beneath the graph of roc_auc
    """
    fpr, tpr, _, idx = roc(y, p, k, thresholds_num)
    return np.trapz(tpr[:, k_pos], x=fpr[:, k_pos])


def _prepare_data(a):
    if len(a.shape) > 1:
        if a.shape[1] > 1:
            a = np.argmax(a, axis=1)
        a = a.reshape((-1,))
    return a
