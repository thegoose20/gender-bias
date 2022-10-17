# Custom functions ordered by usage in `DocumentClassifiers.ipynb`

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def readData(filepath, seperator):
    f = open(filepath, "r")
    text = f.read()
    data_split = text.split(seperator)
    return np.array(data_split)


labels = {
    "Unknown": 0, "Non-binary": 1, "Feminine": 2, "Masculine": 3,
    "Generalization": 4, "Gendered-Pronoun": 5, "Gendered-Role": 6,
    "Occupation": 7, "Omission":8, "Stereotype": 9, "Empowering": 10
         }

def getNumericLabels(target_data, labels_dict=labels):
    numeric_target_data = []
    for target_str in target_data:
        target_list = target_str.split(", ")
        numeric_target = []
        for target in target_list:
            numeric_target += [labels_dict[target]]
        numeric_tuple = tuple(numeric_target)
        numeric_target_data += [numeric_tuple]
    return numeric_target_data


def getPerformanceMetrics(y_test_binarized, predicted, matrix, classes, original_classes, labels_dict=labels):
    tn = matrix[:, 0, 0]
    fn = matrix[:, 1, 0]
    tp = matrix[:, 1, 1]
    fp = matrix[:, 0, 1]
    class_to_name = dict(zip(list(labels.values()), list(labels.keys())))
    class_names = [class_to_name[c] for c in original_classes]
    
    [precision, recall, f_1, support] = precision_recall_fscore_support(
        y_test_binarized, predicted, beta=1.0, 
        zero_division=0, labels=classes
    )
    
    df = pd.DataFrame({
        "labels":class_names, "true_neg":tn, "false_neg":fn, "true_pos":tp, "false_pos":fp,
        "precision":precision, "recall":recall, "f_1":f_1
    })
    
    return df