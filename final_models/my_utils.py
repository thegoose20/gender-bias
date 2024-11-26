import config
import pandas as pd
import joblib
import os, re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
from gensim.models import FastText#, Word2Vec
from gensim.utils import tokenize
from gensim import utils
from gensim.test.utils import get_tmpfile
import sklearn.metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pathlib import Path


'''
*******************
TRAINED/FIT MODELS
*******************
'''

'''
Models for extracting features for classifying tokens with Linguistic (Gendered Pronoun, Gendered Role, and Generalization) labels
'''
ft_model = FastText.load(config.fasttext_path+"fasttext_cbow_100d.model")             # Custom FastText word embeddings of 100 dimensions for lowercased tokens
mlb_ling = joblib.load(config.models_path+"multilabel_token/mlb_targets_ling.joblib")   # Binary encoding for Gendered Pronoun, Gendered Role, and Generalization labels

'''
Models for extracting features for classifying documents with Stereotype and Omission labels
'''
cvectorizer = joblib.load(config.models_path+"multilabel_document/count_vectorizer.joblib")
tfidf = joblib.load(config.models_path+"multilabel_document/tfidf_transformer.joblib")
mlb_so = joblib.load(config.models_path+"multilabel_document/mlb_targets_os.joblib")  # Binary encoding for Stereotype and Omission labels


'''
************************
PREPROCESSING FUNCTIONS
************************
'''

'''
Preprocess data for the multilabel token classifier, inputting a DataFrame, the name
of a column in that DataFrame, and a list of labels that may appear in that column that 
will be the tags the classifier assigns (a.k.a. predictions assigned to input text).  
Output a DataFrame that has one token per row and a unique list of tags associated with
each token.
'''
def preprocessTokenData(df, col, label_list):
    initial_shape = df.shape
    # Change any tags not in label_list to "O"
    df_l = df.loc[df[col].isin(label_list)]
    df_o = df.loc[~df[col].isin(label_list)]
    df_o = df_o.drop(columns=[col])
    df_o.insert(len(df_o.columns), col, (["O"]*(df_o.shape[0])))
    df = pd.concat([df_l, df_o])
    df = df.sort_values(by="token_id")
    assert initial_shape == df.shape, "The DataFrame should have the same number of rows and columns after changing select column values."
    df = df.drop_duplicates()

    # Replace tags with labels, removing "B-" and "I-" from the start of the tags
    old_col = df[col]
    new_col = [tag[2:] if tag != "O" else tag for tag in old_col]
    df = df.drop(columns=[col])
    df.insert((len(df.columns)-2), col, new_col)
    
    # Group by token, so there's one row per token and lists of tags for each token
    df = implodeDataFrame(df, [
        "description_id", "sentence_id", "token_id", "token", "pos", "field", "token_offsets", "fold"
    ])
    df = df.reset_index()
    
    # Deduplicate tag lists and remove any "O" tags from lists with other values
    old_col = list(df[col])
    dedup_col = [list(set(value_list)) for value_list in old_col]
    assert len(old_col) == len(dedup_col), "The column should have the same number of rows."
    new_col = []
    for col_list in dedup_col:
        if ("O" in col_list) and (len(col_list) > 1):
            col_list.remove("O")
        col_list.sort()
        new_col += [col_list]
    assert len(new_col) == len(old_col), "The column should have the same number of rows."
    df = df.drop(columns=[col])
    df.insert((len(df.columns)-2), col, new_col)
    
    return df

'''
Load a data file in comma-separated values format as a pandas DataFrame.
- Function inputs: the file path (as a string) and a boolean value (True or False)
indicating whether or not each row in the data file has a unique identifier.
- Function output: a pandas DataFrame.
'''
def loadCSVData(filepath, has_row_ids):
    df = pd.read_csv(filepath, low_memory=False)
    if not has_row_ids:
        df = df.reset_index()
        df = df.rename(columns={"index":"record_id"})
    return df


'''
Prepare the data for token classification
- Function inputs: a pandas DataFrame, the column name(s) with text for 
classification (as a list of strings), and name of the column with the rows' 
unique identifiers (default provided).  If more than one column name is input, those columns' 
text will be joined to create a new column named text.
- Function output: a pandas DataFrame with one row per token and columns for
row unique identifiers, tokens, and token unique identifiers.
'''
def getTokenDF(df, cols, row_id="record_id"):
    df = df.fillna("") # Replace NaN (empty) values with an empty string
    if len(cols) > 1:
        col_name = "text"
        df[col_name] = df[cols].apply(lambda row: (". ".join(row.values.astype(str))).lstrip(". "), axis=1)
    elif len(cols) == 1:
        col_name = cols[0]
    else: 
        return "Please provide a list of text column names containing at least one column."

    non_empty_df = df.loc[df[col_name] != ""]
    incl_row_ids = list(non_empty_df[row_id])
    
    text = list(non_empty_df[col_name])
    token_col = []
    token_id_col = []
    last_id = 0
    for row in text:
        tokens = word_tokenize(str(row))
        token_ids = list(range(last_id, len(tokens)+last_id))
        last_id = last_id+len(tokens)
        token_id_col += [token_ids]
        token_col += [tokens]
    new_df = pd.DataFrame({"record_id": incl_row_ids, "token_id": token_id_col, "token": token_col})
    new_df = new_df.explode(["token_id", "token"])

    return new_df

'''
Preprocess features from a previous token classifier's predictions by getting
one label per record.
Function inputs: a DataFrame and a column in that DataFrame to flatten (default
provided).
Function output: a DataFrame with a new column containing a 1D list of labels
for each record (document).
'''
def flattenFeatureColumn(df, feature_col="prediction"):
    doc_feature = []
    col_list = list(df[feature_col])
    for row in col_list:
        unique_values = []
        for item in row:
            if item == tuple():
                continue
            elif type(item) == str:
                unique_values = row
            else:
                for subitem in item:
                    if type(subitem) == str:
                        unique_values += [subitem]
        unique_values = list(set(unique_values))
        if "O" in unique_values:
            unique_values.remove("O")
        doc_feature += [unique_values]
    
    df.insert(len(df.columns), "document_prediction", doc_feature)
    
    return df

'''
Associate a token classifier's labels to each document to include those 
classifier-assigned labels as features for document classification.
Function input: a DataFrame of documents to be classified with one row
per document, a DataFrame of classified tokens with one row per token,
the column name of the unique identifiers for the DataFrames' records 
(default provided), and the name of the latter DataFrame's classifier-assigned
labels column (default provided).
Function output: the DataFrame of documents to be classified with a new 
column containing non-repeating lists of labels that each document's tokens
were classified with.
'''
def preprocessClassifiedDocs(doc_df, feature_df, row_id="record_id", pred_col="document_prediction"):
    imploded_df = implodeDataFrame(feature_df, [row_id]).reset_index()
    flattened_df = flattenFeatureColumn(imploded_df)
    to_join = imploded_df[[row_id, pred_col]]
    new_df = to_join.join(doc_df.set_index("record_id"), on="record_id", how="outer")
    return new_df

# ling_df = bt[["record_id", "Title"]]
# bt_clf = bt_tokenized_imploded.join(bt_sub.set_index("record_id"), on="record_id", how="outer")
# bt_clf.head()


'''
*****************************
FEATURE EXTRACTION FUNCTIONS
*****************************
'''

'''
Get features for the multilabel token classifier given an input DataFrame with one 
token per row and, optionally, an embedding model and feature columns (the defaults are the
100-dimension FastText embedding model and the columns 'token_id' and 'token', respectively).
Output a feature matrix as a numpy array.
'''
def getFeatures(df, embedding_model=ft_model, feature_cols=["token_id", "token"]):
    # Zip the features
    feature_data = list(zip(df[feature_cols[0]], df[feature_cols[1]]))
    
    # Make FastText feature matrix
    feature_list = [embedding_model.wv[token.lower()] for token_id,token in feature_data]
    return np.array(feature_list)


'''
Represent text with a TFIDF matrix.
- Function inputs: DataFrame with one row per text entry to be classified, and the name
of the column(s) with text to be classified, a pre-fit CountVectorizer (default provided), 
and a pre-fit TfidfTransformer (default provided).  If more than one column is input, the 
text in those columns will be joined to create a new column named text.
- Function output: a TFIDF (feature) matrix.
'''
def docToTfidf(df, cols, count_vectorizer=cvectorizer, tfidf_transformer=tfidf):
    df = df.fillna("")  # Replace any NaN or empty values with an empty string
    if len(cols) > 1:    
        col_name = "text"
        df[col_name] = df[cols].apply(lambda row: (". ".join(row.values.astype(str))).lstrip(". "), axis=1)
    elif len(cols) == 1:
        col_name = cols[0]
    else:
        return "Please input a non-empty list of strings of DataFrame column names as the second function parameter."

    vectorized = cvectorizer.transform(df[col_name])
    docs = tfidf.transform(vectorized)
    return docs

# '''
# Combine text represented as a TFIDF matrix with additional features.
# - Function inputs: a TFIDF matrix and a feature matrix.
# - Function outputs: a feature matrix (as a sparse SciPy matrix).
# '''
# def combineDocFeatures(docs, features):
#     X = scipy.sparse.hstack([docs, features])
#     return X



'''
********************************
CLASSIFICATION EXPORT FUNCTIONS
********************************
'''

'''
Implode a DataFrame (opposite of df.explode()).
- Function inputs: a DataFrame and a non-empty list of columns by which to group the
rest of the data.
- Function output: a DataFrame where the values in all columns except those included as 
function inputs are aggregated as a list per row (so a cell may have repeated values).
'''
def implodeDataFrame(df, cols_to_groupby):
    cols_to_agg = list(df.columns)
    for col in cols_to_groupby:
        cols_to_agg.remove(col)
    agg_dict = dict.fromkeys(cols_to_agg, lambda x: x.tolist())
    return df.groupby(cols_to_groupby).agg(agg_dict).reset_index().set_index(cols_to_groupby)

'''
Implode a DataFrame (opposite of df.explode()).
- Function inputs: a DataFrame and a non-empty list of columns by which to group the
rest of the data.
- Function output: a DataFrame where the values in all columns except those included as 
function inputs are aggregated as a set per row (so there are NO repeated values in a cell).
'''
def implodeDataFrameUnique(df, cols_to_groupby):
    cols_to_agg = list(df.columns)
    for col in cols_to_groupby:
        cols_to_agg.remove(col)
    agg_dict = dict.fromkeys(cols_to_agg, lambda x: list(set(x)))
    return df.groupby(cols_to_groupby).agg(agg_dict).reset_index().set_index(cols_to_groupby)


'''
Export the classified data.
- Function inputs: the DataFrame with all features input into the classifier, the 
predictions output from the classifier, the MultiLabelBinarizer for transforming
the predictions from binary to text representation, the filepath for where
to save and how to name the classified data, and the column name for record unique
identifiers (default provided).
- Function output: the DataFrame with all features input into the classifier with a
new column for the classifier's labels, with one row per token.
'''
def exportClassifiedData(df, y, mlb, filepath, filename, id_col="record_id"):
    predictions = mlb.inverse_transform(y)
    
    # Transform each prediction into a 1D list of strings
    pred_col = []
    for values in predictions:
        preds = []
        if (values != [tuple()]) and (values != []):
            for t in values:
                if len(t) > 0:
                    label = str(t).strip("(',)")
                    if label not in preds:
                        preds += [label]
        pred_col += [preds]
    df.insert(len(df.columns), "prediction", pred_col)
    
    Path(filepath).mkdir(parents=True, exist_ok=True)  # Create any directories in the filepath that don't exist
    df.to_csv(filepath+filename)
    
    return df


'''
********************************
EVALUATION FUNCTIONS
********************************
'''

'''
Using the expected and predicted DataFrames, create an evaluation DataFrame with a column
recording whether the prediction was a true positive, false positive, false negative, or
true negative.
'''
def getTpTnFpFn(exp_df, pred_df, pred_col, exp_col, no_tag_value, left_on_cols, right_on_cols):
    # Add the predicted tags to the DataFrame with expected tags
    exp_pred_df = pd.merge(
        left=exp_df, 
        right=pred_df, 
        how="outer",
        left_on=left_on_cols,
        right_on=right_on_cols,
        suffixes=["", "_pred"],
        indicator=True
    )

    # Replace any NaN values with "O" to indicate no predicted tag
    exp_pred_df[exp_col] = exp_pred_df[exp_col].fillna(no_tag_value)
    exp_pred_df[pred_col] = exp_pred_df[pred_col].fillna(no_tag_value)

    # Find true negatives based on the expected and predicted tags
    sub_exp_pred_df = exp_pred_df.loc[exp_pred_df[exp_col] == no_tag_value]
    sub_exp_pred_df = sub_exp_pred_df.loc[sub_exp_pred_df[pred_col] == no_tag_value]
    sub_exp_pred_df = sub_exp_pred_df.drop(columns=["_merge"])
    sub_exp_pred_df.insert( len(sub_exp_pred_df.columns), "_merge", ( ["true negative"]*(sub_exp_pred_df.shape[0]) ) )
    # Record false negatives, false positives, and true positives based on the merge values
    sub_exp_pred_df2 = exp_pred_df.loc[~exp_pred_df.index.isin(sub_exp_pred_df.index)]
    sub_exp_pred_df2 = sub_exp_pred_df2.replace(to_replace="left_only", value="false negative")
    sub_exp_pred_df2 = sub_exp_pred_df2.replace(to_replace="right_only", value="false positive")
    sub_exp_pred_df2 = sub_exp_pred_df2.replace(to_replace="both", value="true positive")
    # Combine the DataFrames to include all agreement types and sort the DataFrame
    eval_df = pd.concat([sub_exp_pred_df,sub_exp_pred_df2])
    eval_df = eval_df.sort_index()
    
    return eval_df


'''
Calculate precision, recall, and F1 score based on the input
true positive count, false positive count, and false negative count.
'''
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

'''
Calculate the precision, recall, and F1 scores per label and return a DataFrame
with those scores and the true positive, false positive, and false negative counts
per label.
'''
def getPerformanceScores(eval_df, exp_col, pred_col, labels):
    agmt_scores = pd.DataFrame.from_dict({
            "label":[], "false negative":[], "false positive":[],
            "true positive":[], "precision":[], "recall":[], "f1":[]
        })
    for label in labels:
        agmt_df = pd.concat([eval_df.loc[eval_df[exp_col] == label], eval_df.loc[eval_df[pred_col] == label]])
        agmt_df = agmt_df.drop_duplicates() # True positives will have been duplicated in line above
        tp = agmt_df.loc[agmt_df._merge == "true positive"].shape[0]
        fp = agmt_df.loc[agmt_df._merge == "false positive"].shape[0]
        fn = agmt_df.loc[agmt_df._merge == "false negative"].shape[0]
        prec, rec, f1 = precisionRecallF1(tp, fp, fn)
        label_agmt = pd.DataFrame.from_dict({
                "label":[label], "false negative":[fn], "false positive":[fp],
                "true positive":[tp], "precision":[prec], "recall":[rec], "f1":[f1]
            })
        agmt_scores = pd.concat([agmt_scores, label_agmt])
    return agmt_scores