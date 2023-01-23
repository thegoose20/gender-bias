import pandas as pd
import numpy as np
# For preprocessing the text
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


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