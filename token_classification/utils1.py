# -----------------------------------
# -----------------------------------
# CUSTOM FUNCTIONS FOR EXPERIMENT 1
# -----------------------------------
# -----------------------------------

import utils, config
import os, re
import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim import utils as gensim_utils
from sklearn.preprocessing import MultiLabelBinarizer


# Preprocessing: retrieve custom fastText embedding for input token
# ------------------------
d = 100  # Dimensionality of word embeddings
file_name = config.fasttext_path+"fasttext{}_lowercased.model".format(d)
embedding_model = FastText.load(file_name)
def extractFastTextEmbedding(token, fasttext_model=embedding_model):
    if token.isalpha():
        token = token.lower()
    embedding = fasttext_model.wv[token]
    return embedding

# INPUT:  Train and dev DataFrames of annotations with one row per token-BIO tag pair, 
#         column name, and list of BIO-tags or labels
# OUTPUT: DataFrames of annotations with only labels or BIO tags for input label list (all 
#         other values in that column (`col`) replaced with "O")
def selectDataForLabels(df, col, label_list):
    df_l = df.loc[df[col].isin(label_list)]

    df_o = df.loc[~df[col].isin(label_list)]
    df_o = df_o.drop(columns=[col])
    df_o.insert(len(df_o.columns), col, (["O"]*(df_o.shape[0])))

    df = pd.concat([df_l, df_o])
    df = df.sort_values(by="token_id")
    return df

def getColumnValuesAsLists(df, col_name):
    col_values = list(df[col_name])
    col_list_values = [value[1:-1].split(", ") for value in col_values]
    df[col_name] = col_list_values
    return df


# Preprocessing for Model 1 (Multilabel Classifier for Linguistic Labels)
# --------------------------------------------------------------------------


# Load data
# ------------------------
def loadData(df):
#     df = df.drop(columns=["ann_id"])
    df = df.drop_duplicates()
    # Remove Non-binary labels as these were mistaken labels identified early on that were meant to be excluded, 
    # and because only one token has this label, it prevents the data from being input into the models with cross-validation
    df = df.loc[~df.tag.isin(["B-Nonbinary", "I-Nonbinary"])]
    df = df.drop(columns=["description_id", "field", "subset", "token_offsets"])
    df = utils.implodeDataFrame(df, ["sentence_id", "token_id", "token", "pos"])
    return df.reset_index()

# Zip tokens & feature columns into a list of tuples: [(token_id, token), ... ]
# ------------------------
def zipTokensFeatures(loaded_data, feature_cols=["token_id", "token"]):
    token_data = list(zip(loaded_data[feature_cols[0]], loaded_data[feature_cols[1]]))
    return token_data


# Extract fastText features
# ------------------------
def makeFastTextFeatureMatrix(token_data):
    feature_list = [extractFastTextEmbedding(token) for token_id,token in token_data]
    return np.array(feature_list)


# Binarize targets
# ------------------------
def binarizeTrainTargets(train_data, target_col="tag", mlb=MultiLabelBinarizer()):
    y_train_labels = train_data[target_col]
    y_train = mlb.fit_transform(y_train_labels)
    return mlb, y_train

def binarizeDevTargets(mlb, dev_data, target_col="tag"):
    y_dev_labels = dev_data[target_col]
    y_dev = mlb.transform(y_dev_labels)
    return y_dev

# --------------------------------------------------------------------------


# Preprocessing for Model 2 (Sequence Classifier for Person Name + Occupation Labels)
# --------------------------------------------------------------------------

def zip2FeaturesAndTarget(df, target_col, feature_col1="sentence", feature_col2="pred_ling_tag"):
    feature1_list = list(df[feature_col1])  # sentence
    feature2_list = list(df[feature_col2])  # linguistic label
    tag_list = list(df[target_col])
    length = len(feature1_list)
    return [[tuple((feature1_list[i][j], feature2_list[i][j], tag_list[i][j])) for j in range(len(feature1_list[i]))] for i in range(len(feature1_list))]

def zip1FeatureAndTarget(df, target_col, feature_col1="sentence"):
    feature1_list = list(df[feature_col1])
    tag_list = list(df[target_col])
    length = len(feature1_list)
    return [[tuple((feature1_list[i][j], tag_list[i][j])) for j in range(len(feature1_list[i]))] for i in range(len(feature1_list))]

def extractTokenFeatures(sentence, i):
    token = sentence[i][0]
    if (len(sentence[i]) > 2):
        features = {
        'bias': 1.0,
        'token': token,
        'ling_label':sentence[i][1]
        }
    else:
        features = {
            'bias': 1.0,
            'token': token,
        }
    
    # Add each value in a token's word embedding as a separate feature
    embedding = extractFastTextEmbedding(token)
    for i,n in enumerate(embedding):
        features['e{}'.format(i)] = n
    
    # Record whether a token is the first or last token of a sentence
    if i == 0:
        features['START'] = True
    elif i == (len(sentence) - 1):
        features['END'] = True
    
    return features

def extractSentenceFeatures(sentence):
    return [extractTokenFeatures(sentence, i) for i in range(len(sentence))]

# If multiple tags, only extract the first
def extractSentenceTargets(sentence):
    return [s[-1][0] for s in sentence]

def extractSentenceTokens(sentence):
    return [token for token, ling_label, tag_list in sentence]



def getAnnotationAgreement(df, pred_col, exp_col):
    agmt_types = []  # TP, FP, TN, FN
    agmt_labels = [] # relevant label for agreement type
    exps, preds = list(df[exp_col]), list(df[pred_col])
    rows = df.shape[0]
    for i in range(rows):
        agmt, label = "", ""
        exp, pred = exps[i], preds[i]
        # Remove duplicates from the expected and predicted values
        exp = list(set(exp))
        pred = list(set(pred))
        # If there's only one expected value and one predicted value, compare them
        if (len(exp) == 1) and (len(pred) == 1):
            if (exp[0] == "O"): 
                if (pred[0] == "O"):
                    agmt = "true negative"
                    label = "O"
                else:
                    agmt = "false positive"
                    label = pred[0]
            elif (pred[0] == "O"):
                agmt = "false negative"
                label = exp[0]
            elif (exp[0] == pred[0]):
                agmt = "true positive"
                label = exp[0]
            else:
                agmt = "false positive"
                label = pred[0]
        else:
            # If there is more than one value in expected and predicted lists, compare
            # all their values excluding any "O" values
            if "O" in exp:
                exp.remove("O")
            if "O" in pred:
                pred.remove("O")
            for p in pred:
                # If any of the remaining predicted values match an expceted value, 
                # record true positive agreement
                if p in exp:
                    agmt = "true positive"
                    label = p
                    break
            # In any other cases of label mismatches, record false positive agreement
            if len(agmt) == 0:
                if len(pred) > 0:
                    agmt = "false positive"
                    label = pred[0]
                else:
                    agmt = "false negative"
                    label = exp[0]
        
        assert len(agmt) > 0, "An agreement type should be recorded for every row."
        
        agmt_types += [agmt]
        agmt_labels += [label]

    assert len(agmt_types) == df.shape[0], "There should be one agreement type per row."
    assert len(agmt_types) == len(agmt_labels), "There should be one label associated with each agreement type."
    
    return agmt_types, agmt_labels

def getAnnotationAgreementMetrics(df, category):
    ann_agmts = dict(df["annotation_agreement"].value_counts())
    prec, rec, f1 = utils.precisionRecallF1(
        ann_agmts["true positive"], 
        ann_agmts["false positive"], 
        ann_agmts["false negative"]
    )
    metrics = pd.DataFrame({
        "labels":[category], "false negative":[ann_agmts["false negative"]], 
        "true positive":[ann_agmts["true positive"]], "false positive":[ann_agmts["false positive"]],
        "precision":[prec], "recall":[rec], "f_1":[f1]
    })
    return metrics
# --------------------------------------------------------------------------


# Preprocessing for Model 3 (Document Classifier for Stereotype + Omission Labels)
# --------------------------------------------------------------------------

def flattenFeatureCol(grouped, feature_col):
    doc_feature = []
    col_list = list(grouped[feature_col])
    for row in col_list:
        unique_values = []
        for item in row:
            if type(item) == str:
                unique_values = row
            else:
                for subitem in item:
                    if type(subitem) == str:
                        unique_values += [subitem]
        unique_values = list(set(unique_values))
        if "O" in unique_values:
            unique_values.remove("O")
        doc_feature += [unique_values]
    return doc_feature
