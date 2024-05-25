import config
import pandas as pd
import numpy as np
import re
# For preprocessing the text
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
# For classification
from sklearn.pipeline import Pipeline
# For classifier evaluation
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay#, plot_confusion_matrix 
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score#, accuracy_score, jaccard_score
from intervaltree import Interval, IntervalTree


#################################################################
#################################################################
#################################################################
# Split Data for Token Classification
#################################################################
#################################################################
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
#################################################################
#################################################################
# Sequence & Token Classifiers
#################################################################
#################################################################
#################################################################

def getColumnValuesAsLists(df, col_name):
    col_values = list(df[col_name])
    col_list_values = [value[2:-2].split("', '") for value in col_values]
    df[col_name] = col_list_values
    return df


# INPUT:  Train and dev DataFrames of annotations with one row per token-BIO tag pair, 
#         column name, and list of BIO-tags or labels
# OUTPUT: DataFrames of annotations with only labels or BIO tags for input label list (all 
#         other values in that column (`col`) replaced with "O")
def selectDataForLabels(df_train, df_dev, col, label_list):
    df_train_l = df_train.loc[df_train[col].isin(label_list)]

    df_train_o = df_train.loc[~df_train[col].isin(label_list)]
    df_train_o = df_train_o.drop(columns=[col])
    df_train_o.insert(len(df_train_o.columns), col, (["O"]*(df_train_o.shape[0])))

    df_dev_l= df_dev.loc[df_dev.tag.isin(label_list)]

    df_dev_o = df_dev.loc[~df_dev.tag.isin(label_list)]
    df_dev_o = df_dev_o.drop(columns=[col])
    df_dev_o.insert(len(df_dev_o.columns), col, (["O"]*(df_dev_o.shape[0])))

    df_train = pd.concat([df_train_l, df_train_o])
    df_train = df_train.sort_values(by="token_id")
    df_dev = pd.concat([df_dev_l, df_dev_o])
    df_dev = df_dev.sort_values(by="token_id")
    return df_train, df_dev


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
    df = df.drop_duplicates()
    return df


# INPUT:  list of sentences, where each sentence is a list of tokens (strings)
# OUTPUT: the length of the longest sentence
def getMaxSentenceLength(sentences):
    sent_lengths = [len(s) for s in sentences]
    return np.max(sent_lengths)

# INPUT:  list of sentences, where each sentence is a list of tokens (strings)
# OUTPUT: list of sentences that are all the same length, with the special token 
#         'PAD' added repeatedly to the end of sentences shorter than the 
#         longest sentence
def padSentences(sentences, max_length, pad_token="PAD"):
    new_sentences = []
    for s in sentences:
        pad_count = max_length - len(s)
        padding = [pad_token]*pad_count
        new_s = s + padding
        new_sentences += [new_s]
    return new_sentences

# Replace the input DataFrame's sentence column with padded sentences
def addPaddedSentenceColumn(df, col_name="sentence"):
    sentences = list(df.sentence)
    max_length = getMaxSentenceLength(sentences)
    
    padded_sentences = padSentences(sentences, max_length)
    
    sent_col_i = (list(df.columns)).index(col_name)
    df = df.drop(columns=[col_name])
    df.insert(sent_col_i-1, "sentence", padded_sentences)
    
    return df


# INPUT:  a DataFrame and the name of the target column
# OUTPUT: a list of sentences, where each sentence item is a tuple of three items:
#         a token, the token's part-of-speech tag, and the token's target tag
def zipFeaturesAndTarget(df, target_col):
    sent_list = list(df.sentence)
    pos_list = list(df.pos)
    tag_list = list(df[target_col])
    length = len(sent_list)
    return [[tuple((sent_list[i][j], pos_list[i][j], tag_list[i][j])) for j in range(len(sent_list[i]))] for i in range(len(sent_list))]


# Create an interval tree from the token offsets of the input DataFrame, columns, and tag names
def createIntervalTree(df, offsets_col, tag_col, tag_names):
    subdf = df.loc[df[tag_col].isin(tag_names)]
    offsets_list = list(subdf[offsets_col])
    return IntervalTree.from_tuples(offsets_list)


# Get counts of true positives, false positives, and false negatives 
# for exactly matching, overlapping, and enveloping annotations 
def looseAgreement(tree_exp, tree_pred):
    tp_count, fp_count, fn_count = 0, 0, 0
    # Note: TP will actually be TN when evaluating for 'O' tags 
    for annotation in tree_exp:
        tp_count += len(tree_pred.overlap(annotation))
    fn_count = len(tree_exp.difference(tree_pred))
    fp_count = len(tree_pred.difference(tree_exp))
    return tp_count, fp_count, fn_count
    
    
# Calculate precision, recall, and F1 scores, returning
# 1 in the case of zero division
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


# INPUT:  Prediction DataFrame for a particular tag category, the category name, 
#         agreement type to record, match type to record, expected tag, & predicted tag
# OUTPUT: Input prediction DataFrame with additional columns for 'strict_agreement' 
#         and 'match_type' 
def recordCatAgreementAndMatchType(subdf_pred, category, agmt_type, match_type, exp_value, pred_value):
    subdf_pred_type = subdf_pred.loc[(subdf_pred["tag_cat_{}_expected".format(category)] == exp_value)]
    subdf_pred_type = subdf_pred_type.loc[(subdf_pred_type["tag_cat_{}_predicted".format(category)] == pred_value)]
    strict_agmt_col = [agmt_type]*subdf_pred_type.shape[0]
    match_type_col = [match_type]*subdf_pred_type.shape[0]
    subdf_pred_type.insert(len(subdf_pred_type.columns), "strict_agreement", strict_agmt_col)
    subdf_pred_type.insert(len(subdf_pred_type.columns), "match_type", match_type_col)
    return subdf_pred_type

def addCatAgreementAndMatchTypeCols(df_dev_exploded, category, tags):
    # Get relevant subset of input DataFrame
    subdf_pred = df_dev_exploded[["sentence_id", "token_id", "token", "tag_cat_{}_expected".format(category), "tag_cat_{}_predicted".format(category)]]
    
    # True negatives
    subdf_pred_tn = recordCatAgreementAndMatchType(subdf_pred, category, "TN", "exact_match", "O", "O")
    # True positives
    subdf_pred_tp_b = recordCatAgreementAndMatchType(subdf_pred, category, "TP", "exact_match", tags[0], tags[0])
    subdf_pred_tp_i = recordCatAgreementAndMatchType(subdf_pred, category, "TP", "exact_match", tags[1], tags[1])
    # False positives
    subdf_pred_fp_bi = recordCatAgreementAndMatchType(subdf_pred, category, "FP", "category_match", tags[0], tags[1])
    subdf_pred_fp_ib = recordCatAgreementAndMatchType(subdf_pred, category, "FP", "category_match", tags[1], tags[0])
    subdf_pred_fp_oi = recordCatAgreementAndMatchType(subdf_pred, category, "FP", "mismatch", "O", tags[1])
    subdf_pred_fp_ob = recordCatAgreementAndMatchType(subdf_pred, category, "FP", "mismatch", "O", tags[0])
    # False negatives
    subdf_pred_fn_io = recordCatAgreementAndMatchType(subdf_pred, category, "FN", "mismatch", tags[1], "O")
    subdf_pred_fn_bo = recordCatAgreementAndMatchType(subdf_pred, category, "FN", "mismatch", tags[0], "O")
    
    # Make new a DataFrame
    subdf_pred_new = pd.concat([
        subdf_pred_tn, 
        subdf_pred_tp_b, subdf_pred_tp_i, 
        subdf_pred_fp_bi, subdf_pred_fp_ib, subdf_pred_fp_oi, subdf_pred_fp_ob,
        subdf_pred_fn_io, subdf_pred_fn_bo
    ])
    assert subdf_pred.shape[0] == subdf_pred_new.shape[0]
    subdf_pred_new[subdf_pred_new.match_type == "category_match"]
    subdf_pred_new = subdf_pred_new.sort_values(by=["token_id"])
    
    return subdf_pred_new



# Since multiple tags are possibly correct for certain tokens, record a prediction as 
# correct if one of the expected tags matches the predicted tag
def isPredictedInExpected(df, exp_col, pred_col, agmt_col_name, no_tag_value):
    expected_labels = list(df[exp_col])
    predicted_labels = list(df[pred_col])
    rows = df.shape[0]
    _merge = []
    for i in range(rows):
        exp, pred = expected_labels[i], predicted_labels[i]
        if (no_tag_value in exp) and (pred == no_tag_value):
            _merge += ["true negative"]
        elif pred in exp:
            _merge += ["true positive"]
        else: # if (not pred in exp):
            if (pred == no_tag_value):
                _merge += ["false negative"]
            else:
                _merge += ["false positive"]            
    assert len(_merge) == rows, "There should be one agreement type per row."
    df.insert(len(df.columns), agmt_col_name, _merge)
    return df


def getScoresByCatTags(df, eval_col, tag, exp_col, pred_col, id_col):
    # Filter out the true negatives (rows with "O" tag as expected and predicted values)
    subdf = df.loc[df[eval_col] != "true negative"]
    
    # Find true positives and false positives
    positives = subdf.loc[subdf[pred_col] == tag]
    positives_dict = positives[eval_col].value_counts().to_dict()
    if "true positive" in positives_dict:
        tp = positives_dict["true positive"]
    else:
        tp = 0
    if "false positive" in positives_dict:
        fp = positives_dict["false positive"]
    else:
        fp = 0
    
    # Find false negatives
    negatives = subdf.loc[subdf[pred_col] == "O"]
    exp_tag_lists = list(negatives[exp_col])
    fn = 0
    for tag_list in exp_tag_lists:
        if (not ("O" in tag_list)):
            if (tag in tag_list):
                fn += 1
    
    # Calculate precision, recall, and f1 score (in case of zero division, return 0)
    prec, rec, f1 = precisionRecallF1(tp, fp, fn)
    
    return pd.DataFrame.from_dict({
        "tag(s)":[tag], "false negative":[fn], "false positive":[fp], 
         "true positive":[tp], "precision":[prec], "recall":[rec], "f1":[f1]
    })



def makePredictionDF(predictions, dev_data, exp_col_name, pred_col_name, no_tag_value, mlb):
    pred_labels = mlb.inverse_transform(predictions)
    pred_df = dev_data.drop(columns=[exp_col_name])
    pred_df.insert(len(pred_df.columns), pred_col_name, pred_labels)
    pred_df = pred_df.explode([pred_col_name])
    pred_df[pred_col_name] = pred_df[pred_col_name].fillna(no_tag_value)
    return pred_df


def makeEvaluationDataFrame(exp_df, pred_df, left_on_cols, right_on_cols, final_cols, exp_col, pred_col, id_col, no_tag_value):
    # Add the predicted tags to the DataFrame with expected tags
    exp_pred_df = pd.merge(
        left=exp_df, right=pred_df, how="outer",
        left_on=left_on_cols,
        right_on=right_on_cols,
        suffixes=["", "_pred"],
        indicator=True
    )
    exp_pred_df = exp_pred_df[final_cols]
    
    # Replace any NaN values with "O" to indicate to predicted label/tag
    exp_pred_df[exp_col] = exp_pred_df[exp_col].fillna("O")
    exp_pred_df[pred_col] = exp_pred_df[pred_col].fillna("O")
    
    # Find true negatives based on the expected and predicted tags
    sub_exp_pred_df = exp_pred_df.loc[exp_pred_df[exp_col] == no_tag_value]
    sub_exp_pred_df = sub_exp_pred_df.loc[sub_exp_pred_df[pred_col] == no_tag_value]
    sub_exp_pred_df.replace(to_replace="both", value="true negative", inplace=True)
    tn_tokens = list(sub_exp_pred_df["token_id"])
    # Record false negatives, false positives, and true positives based on the merge values
    sub_exp_pred_df2 = exp_pred_df.loc[~exp_pred_df["token_id"].isin(tn_tokens)]
    sub_exp_pred_df2 = sub_exp_pred_df2.replace(to_replace="left_only", value="false negative")
    sub_exp_pred_df2 = sub_exp_pred_df2.replace(to_replace="right_only", value="false positive")
    sub_exp_pred_df2 = sub_exp_pred_df2.replace(to_replace="both", value="true positive")
    # Combine the DataFrames to include all agreement types and sort the DataFrame
    eval_df = pd.concat([sub_exp_pred_df,sub_exp_pred_df2])
    eval_df = eval_df.sort_index()
    return eval_df


def getScoresByTags(df, eval_col, tags, exp_col="expected_tag", pred_col="predicted_tag"):
    subdf1 = df.loc[df[pred_col].isin(tags)]
    subdf2 = df.loc[df[exp_col].isin(tags)]
    subdf = pd.concat([subdf1, subdf2])
    tp = subdf.loc[subdf[eval_col] == "true positive"].shape[0]
    tn = subdf.loc[subdf[eval_col] == "true negative"].shape[0]
    fp = subdf.loc[subdf[eval_col] == "false positive"].shape[0]
    fn = subdf.loc[subdf[eval_col] == "false negative"].shape[0]
    # Precision Score: ability of classifier not to label a sample that should be negative as positive; best possible = 1, worst possible = 0
    if (tp+fp) > 0:
        prec = tp/(tp+fp)
    else:
        prec = 0
    # Recall Score: ability of classifier to find all positive samples; best possible = 1, worst possible = 0
    if (tp+fn) > 0:
        rec = tp/(tp+fn)
    else:
        rec = 0
    # F1 Score: harmonic mean of precision and recall; best possible = 1, worst possible = 0
    if (prec+rec) > 0:
        f1 = (2*(prec*rec))/(prec+rec)
    else:
        f1 = 0
    if len(tags) > 1:
        tags = ", ".join(tags)
    return pd.DataFrame.from_dict({
        "tag(s)":tags, "false negative":[fn], "false positive":[fp], "true negative":[tn], 
         "true positive":tp, "precision":[prec], "recall":[rec], "f1":[f1]
    })


def compareExpectedPredicted(loose_eval_df, agmt_col_name, no_tag_value):
    expected_labels = list(loose_eval_df.expected_tag)
    predicted_labels = list(loose_eval_df.predicted_tag)
    rows = loose_eval_df.shape[0]
    _merge = []
    for i in range(rows):
        exp, pred = expected_labels[i], predicted_labels[i]
        if (exp == no_tag_value) and (pred == no_tag_value):
            _merge += ["true negative"]
        elif exp == pred:
            _merge += ["true positive"]
        elif (exp == no_tag_value) and (pred != no_tag_value):
            _merge += ["false_positive"]
        elif (exp != no_tag_value) and (pred == no_tag_value):
            _merge += ["false negative"]
    assert len(_merge) == rows, "There should be one agreement type per row."
    loose_eval_df.insert(len(loose_eval_df.columns), agmt_col_name, _merge)
    return loose_eval_df


def getAgreementStatsForAllTags(eval_df, agmt_col, id_col, tag_col, y_dev, predictions):
    agmt_stats = eval_df[[agmt_col, id_col]].groupby(agmt_col).count().reset_index()
    agmt_stats = agmt_stats.rename(columns={agmt_col:tag_col, id_col:"all"})
    agmt_stats = agmt_stats.set_index(tag_col)
    agmt_stats = agmt_stats.T
    precision = metrics.precision_score(y_dev, predictions, average="macro", zero_division=0)
    recall = metrics.recall_score(y_dev, predictions, average="macro", zero_division=0)
    f1 = metrics.f1_score(y_dev, predictions, average="macro", zero_division=0)
    # jaccard = metrics.jaccard_score(y_dev, predictions, average="macro", zero_division=0)
    metrics_df = pd.DataFrame.from_dict({tag_col: "all", "precision": [precision], "recall": [recall], "f1": [f1]}) #, "jaccard":[jaccard]})
    metrics_df = metrics_df.set_index(tag_col)
    agmt_stats = agmt_stats.join(metrics_df)
    agmt_stats = agmt_stats.reset_index()
    agmt_stats = agmt_stats.rename(columns={"index":tag_col})
    return agmt_stats



def getAnnotationAgreement(df, pred_col, exp_col):
    agmt_types = []  # TP, FP, TN, FN
    exps, preds = list(df[exp_col]), list(df[pred_col])
    rows = df.shape[0]
    for i in range(rows):
        agmt = ""
        exp, pred = exps[i], preds[i]
        # Remove duplicates from the expected and predicted values
        exp = list(set(exp))
        pred = list(set(pred))
        # If there's only one expected value and one predicted value, compare them
        if (len(exp) == 1) and (len(pred) == 1):
            if (exp[0] == "O"): 
                if (pred[0] == "O"):
                    agmt = "true negative"
                else:
                    agmt = "false positive"
            elif (pred[0] == "O"):
                agmt = "false negative"
            elif (exp[0] == pred[0]):
                agmt = "true positive"
            else:
                agmt = "false positive"
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
                    break
            # In any other cases of label mismatches, record false positive agreement
            if len(agmt) == 0:
                agmt = "false positive"
        
        assert len(agmt) > 0, "An agreement type should be recorded for every row."
        
        agmt_types += [agmt]

    assert len(agmt_types) == df.shape[0], "There should be one agreement type per row."
    
    return agmt_types


def getAnnotationAgreementMetrics(df, category):
    ann_agmts = dict(df["annotation_agreement"].value_counts())
    prec, rec, f1 = precisionRecallF1(
        ann_agmts["true positive"], 
        ann_agmts["false positive"], 
        ann_agmts["false negative"]
    )
    metrics = pd.DataFrame({
        "labels":[category], 
        "true negative":[ann_agmts["true negative"]], "false negative":[ann_agmts["false negative"]], 
        "true positive":[ann_agmts["true positive"]], "false positive":[ann_agmts["false positive"]],
        "precision":[prec], "recall":[rec], "f_1":[f1]
    })
    return metrics


#################################################################
#################################################################
#################################################################
# Word Embeddings
#################################################################
#################################################################
#################################################################

def createEmbeddingDataFrame(df, embedding_dict, embedding_col_name, d):
    tokens = list(df.token)
    embedding_list = []
    for token in tokens:
        token = token.lower()
        word_list = re.findall("[a-z]+", token)
        if len(word_list) == 1:
            try:
                embedding = embedding_dict[word_list[0]]
            except KeyError:
                embedding = np.zeros((d,))
            embedding_list += [embedding]
        else:
            embedding_list += [[]]
    new_df = df[["token_id", "token"]]
    new_df.insert(len(new_df.columns)-1, embedding_col_name, embedding_list)
    return new_df