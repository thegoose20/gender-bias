# For path variables
import config, utils
# For data analysis
import pandas as pd
import numpy as np
import os, re
# For creating directories
from pathlib import Path
# For fastText embeddings
from gensim.models import FastText
from gensim import utils as gensim_utils
# For classification
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
# For statistical significance testing
from scipy.stats import ttest_ind


# ------------------------
# FUNCTIONS
# ------------------------
# INPUT:  path to model input data for token classification
# OUTPUT: DataFrame with one row per token
def preprocess(data_path):
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop(columns=["ann_id"])
    df = df.drop_duplicates()
    # Remove Non-binary labels as these were mistaken labels identified early on that were meant to be excluded,
    # and because only one token has this label, it prevents the data from being input into the models with cross-validation
    df = df.loc[df.tag != "B-Nonbinary"]
    df = df.loc[df.tag != "I-Nonbinary"]
    df = df.drop(columns=["description_id", "field", "subset", "token_offsets"])
    df = implodeDataFrame(df, ["sentence_id", "token_id", "token", "pos"])
    return df.reset_index()

# Retrieve pre-trained GloVe embeddings of input dimensions
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


# ------------------------
# SETUP
# ------------------------
train_data_path = config.tokc_path+"model_input/token_train.csv"
train_data = utils.preprocess(train_data_path)
dev_data_path = config.tokc_path+"model_input/token_validate.csv"
dev_data_full = utils.preprocess(dev_data_path)
# Word embeddings' dimensions
dimensions = ["50", "100", "200", "300"]
d = dimensions[0]
# Sample size
# frac_samples = 0.1
n = 200
# Number of times to run classifiers over samples
n_classifiers = 1000

mlb = MultiLabelBinarizer()
target_col = "tag"
feature_cols = ["token_id", "token"]

# Load data
def zipTokensFeatures(loaded_data):
    token_data = list(zip(loaded_data[feature_cols[0]], loaded_data[feature_cols[1]]))
    return token_data

# Extract GloVe features
glove = utils.getGloveEmbeddings(d)
def extractGloveEmbedding(token, embedding_dict=glove, dimensions=int(d)):
    if token.isalpha():
        token = token.lower()
    try:
        embedding = embedding_dict[token]
    except KeyError:
        embedding = np.zeros((dimensions,))
    return embedding.reshape(-1,1)

def makeGloveFeatureMatrix(token_data, dimensions=int(d)):
    feature_list = [extractGloveEmbedding(token) for token_id,token in token_data]
    return np.array(feature_list).reshape(-1,dimensions)

# Extract fastText features
file_name = config.fasttext_path+"fasttext{}_lowercased.model".format(d)
embedding_model = FastText.load(file_name)
def extractFastTextEmbedding(token, fasttext_model=embedding_model):
    if token.isalpha():
        token = token.lower()
    embedding = fasttext_model.wv[token]
    return embedding

def makeFastTextFeatureMatrix(token_data):
    feature_list = [extractFastTextEmbedding(token) for token_id,token in token_data]
    return np.array(feature_list)

# Binarize targets
def binarizeTrainTargets(train_data):
    y_train_labels = train_data[target_col]
    y_train = mlb.fit_transform(y_train_labels)
    return mlb, y_train

def binarizeDevTargets(mlb, dev_data):
    y_dev_labels = dev_data[target_col]
    y_dev = mlb.transform(y_dev_labels)
    return y_dev

# ------------------------
# TRAIN
# ------------------------
# Load and preprocess training data
train_tokens = zipTokensFeatures(train_data)
# Get GloVe features
X_train_glove = makeGloveFeatureMatrix(train_tokens)
# Get custom fastText features
X_train_ft = makeFastTextFeatureMatrix(train_tokens)
# Get targets
mlb, y_train = binarizeTrainTargets(train_data)
# Train a model with GloVe embeddings as features
clf_glove = ClassifierChain(classifier = RandomForestClassifier(random_state=22))
clf_glove.fit(X_train_glove, y_train)
# Train a model with custom fastText embeddings as features
clf_ft = ClassifierChain(classifier = RandomForestClassifier(random_state=22))
clf_ft.fit(X_train_ft, y_train)
print("classifiers trained")

# ------------------------
# PREDICT & EVALUATE
# ------------------------
glove_f1_scores, glove_precision_scores, glove_recall_scores = [], [], []
ft_f1_scores, ft_precision_scores, ft_recall_scores = [], [], []
for n in range(n_classifiers):
    # Load and preprocess a sample of devtest data
    dev_data = dev_data_full.sample(n=n, replace=True)
    dev_tokens = zipTokensFeatures(dev_data)

    # Extract GloVe and custom fastText features for the devtest data sample
    X_dev_glove = makeGloveFeatureMatrix(dev_tokens)
    X_dev_ft = makeFastTextFeatureMatrix(dev_tokens)

    # Get targets
    y_dev = binarizeDevTargets(mlb, dev_data)

    # Predict and evaluate the model with GloVe embeddings as features
    predictions_glove = clf_glove.predict(X_dev_glove)
    glove_precision_scores += metrics.precision_score(y_dev, predictions_glove, average="macro", zero_division=0)
    glove_recall_scores += metrics.recall_score(y_dev, predictions_glove, average="macro", zero_division=0)
    glove_f1_scores += metrics.f1_score(y_dev, predictions_glove, average="macro", zero_division=0)

    # Save the scores
    with open("glove_f1.txt", "a") as f:
        f.write(glove_f1_score)
        f.write("\n")
        f.close()
    with open("glove_precision.txt", "a") as f:
        f.write(glove_precision_score)
        f.write("\n")
        f.close()
    with open("glove_recall.txt", "a") as f:
        f.write(glove_recall_score)
        f.write("\n")
        f.close()

    # Predict and evaluate the model with custom fastText embeddings as features
    predictions_ft = clf_ft.predict(X_dev_ft)
    ft_precision_score = metrics.precision_score(y_dev, predictions_ft, average="macro", zero_division=0)
    ft_recall_score = metrics.recall_score(y_dev, predictions_ft, average="macro", zero_division=0)
    ft_f1_score = metrics.f1_score(y_dev, predictions_ft, average="macro", zero_division=0)

    # Save the scores
    with open("custom_fasttext_f1.txt", "a") as f:
        f.write(ft_f1_score)
        f.write("\n")
        f.close()
    with open("custom_fasttext_precision.txt", "a") as f:
        f.write(ft_precision_score)
        f.write("\n")
        f.close()
    with open("custom_fasttext_recall.txt", "a") as f:
        f.write(ft_recall_score)
        f.write("\n")
        f.close()
    
    print("six files appended to")

# ------------------------
# Compute p-value
# ------------------------
# ttest_precision = ttest_ind(ft_precision_scores, glove_precision_scores)
# ttest_recall = ttest_ind(ft_recall_scores, glove_recall_scores)
# ttest_f1 = ttest_ind(ft_f1_scores, glove_f1_scores)
#
# ttest_precision2 = ttest_ind(glove_precision_scores, ft_precision_scores)
# ttest_recall2 = ttest_ind(glove_recall_scores, ft_recall_scores)
# ttest_f12 = ttest_ind(glove_f1_scores, ft_f1_scores)
