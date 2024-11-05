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
mlb_ling = joblib.load(config.models_path+"multilabel_token/mlb_linglabels.joblib")   # Binary encoding for Gendered Pronoun, Gendered Role, and Generalization labels

'''
Models for extracting features for classifying documents with Stereotype and Omission labels
'''
cvectorizer = joblib.load(config.models_path+"multilabel_document/count_vectorizer.joblib")
tfidf = joblib.load(config.models_path+"multilabel_document/tfidf_transformer.joblib")
mlb_so = joblib.load(config.models_path+"multilabel_document/mlb_targets_so.joblib")  # Binary encoding for Stereotype and Omission labels


'''
************************
PREPROCESSING FUNCTIONS
************************
'''

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
Encode features as 0s and 1s with a pre-fit multilabel binarizer.
- Function inputs: pre-fit MultiLabelBinarizer (already loaded using joblib, not the 
filepath), a DataFrame, and the name of the column containing the text (features)
to encode.
- Function output: a feature matrix.
'''
def binarizeFeatures(mlb, df, feature_col):
    features = mlb.transform(df[feature_col])
    return features

'''
Represent tokens with word embeddings.
- Function inputs: embedding model (already loaded using joblib, not the filepath), DataFrame
with one row per token and columns for tokens and token identifiers, the name of the token
identifier column, and the name of the token column.
- Function outputs: a feature matrix (as a numpy array).
'''
def tokensToEmbeddings(embedding_model, df, id_col, text_col):
    feature_data = list(zip(df[id_col], df[text_col]))
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

'''
Combine text represented as a TFIDF matrix with additional features.
- Function inputs: a TFIDF matrix and a feature matrix.
- Function outputs: a feature matrix (as a sparse SciPy matrix).
'''
def combineDocFeatures(docs, features):
    X = scipy.sparse.hstack([docs, features])
    return X



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
function inputs are aggregated (as a list per row).
'''
def implodeDataFrame(df, cols_to_groupby):
    cols_to_agg = list(df.columns)
    for col in cols_to_groupby:
        cols_to_agg.remove(col)
    agg_dict = dict.fromkeys(cols_to_agg, lambda x: x.tolist())
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