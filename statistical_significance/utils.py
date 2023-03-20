import numpy as np
import pandas as pd
import re
import config

# Calculate precision, recall, and F1 scores from true positive, false positive, and false negative counts,
# returning 0 in the case of zero division
def precisionRecallF1(tp_count, fp_count, fn_count):
    # Precision Score: ability of classifier not to label a sample that should be negative as positive; best possible = 1, worst possible = 0
    if tp_count+fp_count == 0:
        precision = 0
    else:
        precision = (tp_count/(tp_count+fp_count))
    # Recall Score: ability of classifier to find all positive samples; best possible = 1, worst possible = 0
    if tp_count+fn_count == 0:
        recall = 0
    else:
        recall = (tp_count/(tp_count+fn_count))
    # F1 Score: harmonic mean of precision and recall; best possible = 1, worst possible = 0
    if (precision+recall == 0):
        f_1 = 0
    else:
        f_1 = (2*precision*recall)/(precision+recall)
    return precision, recall, f_1


# Opposite of explode - combine rows in all columns not in `cols_to_groupby`, turning those rows values into a list of values
def implodeDataFrame(df, cols_to_groupby):
    cols_to_agg = list(df.columns)
    for col in cols_to_groupby:
        cols_to_agg.remove(col)
    agg_dict = dict.fromkeys(cols_to_agg, lambda x: x.tolist())
    return df.groupby(cols_to_groupby).agg(agg_dict).reset_index().set_index(cols_to_groupby)


# INPUT:  path to model input data for token classification
# OUTPUT: DataFrame with one row per token
def preprocess(data_path):
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop(columns=["ann_id"])
    df = df.drop_duplicates()
    # Remove Non-binary labels as these were mistaken labels identified early on that were meant to be excluded, 
    # and because only one token has this label, it prevents the data from being input into the models with cross-validation
    df = df.loc[df.tag != "B-Nonbinary"]
    df = df.drop(columns=["description_id", "field", "subset", "token_offsets"])
    df = implodeDataFrame(df, ["sentence_id", "token_id", "token", "pos"])
    return df.reset_index()


'''
Functions for GloVe Embeddings
'''

def getGloveEmbeddings(dimensions):
    glove_path = config.inf_data_path+"glove.6B/glove.6B.{}d.txt".format(dimensions)
    glove = dict()
    with open(glove_path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove[word] = vector
    return glove