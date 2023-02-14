import config
import pandas as pd
import numpy as np
import re
# For preprocessing the text
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, f1_score


#################################################################
# Split Data for Token Classification
#################################################################

# INPUT: list of strings (descriptions), list of ids for those strings, list of start offsets for those strings, 
#        list of end offsets for those strings (offsets in the brat rapid annotation tool's standoff format)
# OUTPUT: two dictionaries, both with ids as keys, and one with lists of sentences as values and the other 
#         with lists of those sentences' offsets as values
def getSentsAndOffsetsFromStrings(list_of_strings, list_of_ids, list_of_start_offsets, list_of_end_offsets):
    sents_dict = dict.fromkeys(list_of_ids)
    offsets_dict = dict.fromkeys(list_of_ids)
    j, maxJ = 0, len(list_of_strings)
    while j < maxJ:
        # Get the description's ID
        desc_id = list_of_ids[j]
        # Get the start and end offsets of the description
        desc_start_offset = list_of_start_offsets[j]
        desc_end_offset = list_of_end_offsets[j]
        # Get the description string and its sentences
        desc = list_of_strings[j]
        sents = sent_tokenize(desc)
        # Get the offsets of every sentence
        offsets = []
        for sent in sents:
            start_offset = desc.index(sent)
            end_offset = start_offset + len(sent) + 1
            assert end_offset <= desc_end_offset
            offsets += [tuple((start_offset, end_offset))]
        offsets_dict[desc_id] = offsets
        sents_dict[desc_id] = sents
        
        j += 1
    
    return sents_dict, offsets_dict


# Do the opposite of DataFrame.explode(), creating one row with for each
# value in the cols_to_groupby (list of one or more items) and lists of 
# values in the other columns, and setting the cols_to_groupby as the Index
# or MultiIndex in the resulting DataFrame
def implodeDataFrame(df, cols_to_groupby):
    cols_to_agg = list(df.columns)
    for col in cols_to_groupby:
        cols_to_agg.remove(col)
    agg_dict = dict.fromkeys(cols_to_agg, lambda x: x.tolist())
    return df.groupby(cols_to_groupby).agg(agg_dict).reset_index().set_index(cols_to_groupby)


# INPUT:  DataFrame, fraction of DF to shuffle, and random_state of shuffle
#         Note 1 - fraction defaults to 1 to shuffle the entire DataFrame; 
#                 provide a value <1 to return that fraction of the DataFrame shuffled
#         Note 2 -random_state_value defaults to 7 for reproducibility
# OUTPUT: DataFrame with its rows shuffled
def shuffleDataFrame(df, fraction=1, random_state_value=7):
    return df.sample(frac=fraction, random_state=random_state_value)


# INPUT:  A shuffled DataFrame for a particular metadata field
# OUTPUT: The number of rows from the DataFrame to assign to train, validate (dev), 
#         and (blind) test sets of data f
def getTrainValTestSizes(df):
    indeces = list(df.index)
    
    train = indeces[ : int(df.shape[0]*0.6) ]
    validate = indeces[ int(df.shape[0]*0.6) : (int(df.shape[0]*0.6) + round(df.shape[0]*0.2)) ]
    test = indeces[ (int(df.shape[0]*0.6) + round(df.shape[0]*0.2)) : ]

    return len(train), len(validate), len(test)


# Add a column to the input DataFrame that assigns each row to train, dev, and test
# using the three input sizes
def assignSubsets(df, train_size, validate_size, test_size):
    subset_col = ["train"]*train_size + ["dev"]*validate_size + ["test"]*test_size
    df.insert(len(df.columns)-1, "subset", subset_col)
    return df


# Concatenate the rows assigned to each subset to create one DataFrame each for 
# training, validation, and testing: 
def concatBySubset(df_list, subset):
    df_all = pd.DataFrame()
    for df in df_list:
        df_subset = df.loc[df["subset"] == subset]
        df_all = pd.concat([df_all, df_subset], axis=0)
    return df_all

metadata_fields = ['Biographical / Historical', 'Title', 'Scope and Contents', 'Processing Information']
def getShuffledSplitData(df, field_names=metadata_fields):
    df_bh = df.loc[df.field == field_names[0]]
    df_t = df.loc[df.field == field_names[1]]
    df_sc = df.loc[df.field == field_names[2]]
    df_pi = df.loc[df.field == field_names[3]]
    
    # Shuffle the DataFrames for each metadata field type
    df_bh_shuffled = shuffleDataFrame(df_bh)
    df_t_shuffled = shuffleDataFrame(df_t)
    df_sc_shuffled = shuffleDataFrame(df_sc)
    df_pi_shuffled = shuffleDataFrame(df_pi)
    
    # Get the indeces of rows to assign to train, dev, and test
    train_bh, validate_bh, test_bh = getTrainValTestSizes(df_bh_shuffled)
    assert train_bh+validate_bh+test_bh == df_bh_shuffled.shape[0]
    train_t, validate_t, test_t = getTrainValTestSizes(df_t_shuffled)
    assert train_t+validate_t+test_t == df_t_shuffled.shape[0]
    train_sc, validate_sc, test_sc = getTrainValTestSizes(df_sc_shuffled)
    assert train_sc+validate_sc+test_sc == df_sc_shuffled.shape[0]
    train_pi, validate_pi, test_pi = getTrainValTestSizes(df_pi_shuffled)
    assert train_pi+validate_pi+test_pi == df_pi_shuffled.shape[0]
    
    df_bh = assignSubsets(df_bh_shuffled, train_bh, validate_bh, test_bh)
    df_t = assignSubsets(df_t_shuffled, train_t, validate_t, test_t)
    df_sc = assignSubsets(df_sc_shuffled, train_sc, validate_sc, test_sc)
    df_pi = assignSubsets(df_pi_shuffled, train_pi, validate_pi, test_pi)
    dfs = [df_bh, df_t, df_sc, df_pi]
    
    # Concatenate the rows assigned to each subset to create one DataFrame each for training, validation, and testing: 
    train = concatBySubset(dfs, "train")
    assert train.subset.unique()[0] == "train"

    validate = concatBySubset(dfs, "dev")
    assert validate.subset.unique()[0] == "dev"

    test = concatBySubset(dfs, "test")
    assert test.subset.unique()[0] == "test"

    return train, validate, test

#################################################################
# Baseline Token Classifiers
#################################################################
labels = {
    "Unknown": 0, "Nonbinary": 1, "Feminine": 2, "Masculine": 3,
    "Generalization": 4, "Gendered-Pronoun": 5, "Gendered-Role": 6,
    "Occupation": 7, "Omission":8, "Stereotype": 9, "Empowering": 10
         }

def getNumericLabels(target_data, labels_dict=labels):
    numeric_target_data = []
    for target_str in target_data:
        # If there aren't any labels, add an empty tuple
        if target_str == "":
            numeric_target_data += [tuple(())]
        else:
            target_list = target_str.split(", ")
            numeric_target = []
            for target in target_list:
                numeric_target += [labels_dict[target]]
            numeric_tuple = tuple((numeric_target))
            numeric_target_data += [numeric_tuple]
    return numeric_target_data


def getPerformanceMetrics(y_test_binarized, predicted, matrix, classes, original_classes, no_to_label_dict):
    tn = matrix[:, 0, 0]  # True negatives
    fn = matrix[:, 1, 0]  # False negatives
    tp = matrix[:, 1, 1]  # True positives
    fp = matrix[:, 0, 1]  # False positives
    class_names = [no_to_label_dict[c] for c in original_classes]
    
    [precision, recall, f_1, suport] = precision_recall_fscore_support(
        y_test_binarized, predicted, beta=1.0, zero_division=0, labels=classes
    )
    
    df = pd.DataFrame({
        "labels":class_names, "true_neg":tn, "false_neg":fn, "true_pos":tp, "false_pos":fp,
        "precision":precision, "recall":recall, "f_1":f_1
    })
    
    return df


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


def getEmbeddingsForTokens(embedding_dict, tokens):
    embedding_list = []
    for token in tokens:
        token = token.lower()
        word_list = re.findall("[a-z]+", token)
        if len(word_list) == 1:
            try:
                embedding = embedding_dict[word_list[0]]
            except KeyError:
                embedding = np.array([])
            embedding_list += [embedding]
        else:
            embedding_list += [np.array([])]
    return embedding_list


label_to_cat = {
    "Unknown":"Person-Name", "Nonbinary":"Person-Name", "Feminine":"Person-Name", "Masculine":"Person-Name",
    "Gendered-Pronoun":"Linguistic", "Gendered-Role":"Linguistic", "Generalization":"Linguistic", 
    "Empowering":"Contextual", "Occupation":"Contextual", "Stereotype":"Contextual", "Omission":"Contextual"
}
def addCategoryTagColumn(df, cat_dict=label_to_cat):
    label_tags = list(df.tag)
    category_tags = []
    for label_tag in label_tags:
        if label_tag == "O":
            category_tags += ["O"]
        else:
            label = label_tag[2:]
            category = cat_dict[label]
            category_tag = label_tag[:2]+category
            category_tags += [category_tag]
    df.insert(len(df.columns), "tag_cat", category_tags)
    return df


#################################################################
# Word Embeddings
#################################################################

def createEmbeddingDataFrame(df, embedding_dict, embedding_col_name):
    tokens = list(df.token)
    embedding_list = []
    for token in tokens:
        token = token.lower()
        word_list = re.findall("[a-z]+", token)
        if len(word_list) == 1:
            embedding = embedding_dict[word_list[0]]
            embedding_list += [embedding]
        else:
            embedding_list += [[]]
    new_df = df[["token_id", "token"]]
    new_df.insert(len(new_df.columns)-1, embedding_col_name, embedding_list)
    return new_df