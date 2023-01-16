###############################################################
# CUSTOM FUNCTIONS ORDERED BY USAGE
###############################################################


import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


###############################################################
# Functions for `DocumentClassifiers.ipynb`
###############################################################

def readData(filepath, seperator):
    f = open(filepath, "r")
    text = f.read()
    data_split = text.split(seperator)
    return np.array(data_split)


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



###############################################################
# Functions for SplitData_DocumentClassification.ipynb
###############################################################

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


# Write the documents (each document is one description) to a file, separated
# by a newline character, a pipe, and a newline character: `\n|\n`
def writeDocs(docs, filename, directory="clf_data/"):
    filepath = directory+filename
    f = open(filepath, "a")
    for i,doc in enumerate(docs):    
        doc = doc.strip()                       # Remove leading and trailing whitespace
        if i < len(docs) - 1:
            f.write(doc+"\n|\n")
        else:
            f.write(doc)
    f.close() 
    print("Your documents file has been written!")
    
    
# Write the labels to a file (one row of labels per description), separated 
# by a newline character: `\n`
def writeLabels(labels, filename, directory="clf_data/"):
    filepath = directory+filename
    f = open(filepath, "a")
    for i,label_set in enumerate(labels):    
        label_names = str(label_set)              # Change data type to string
        label_names = label_names[1:-1]           # Remove curly braces
        label_names = label_names.replace("'","") # Remove single quotes surounding each label name
        if i < len(labels) - 1:
            f.write(label_names+"\n")
        else:
            f.write(label_names)
    f.close()
    print("Your labels file has been written!")