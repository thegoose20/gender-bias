import pandas as pd
# For preprocessing the text
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

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
