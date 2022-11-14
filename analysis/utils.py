import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import re
from thefuzz import fuzz, process

fem_patterns = ["wom.n", "girl", "^gal", "female", "lady", "ladies", "wi[fv]e", "her", "she"]
mas_patterns = ["^man", "^men", "boy", "male", "lad$", "lads", "laddie", "husband", "his", "him", "he"]
def addAssociatedGenders(df, fem=fem_patterns,mas=mas_patterns):
    if ("note" in df.columns):
        notes = list(df.note)
        notes = [str(note).lower() for note in notes]  # Turn any NaN values into type string
    texts = list(df.text)
    texts = [text.lower() for text in texts]
    
    genders = []
    for i in range(len(texts)):
        if ("note" in df.columns):
            n = notes[i]
        t = texts[i]
        feminine, masculine = False, False

        for f in fem:
            if feminine:
                break
            if (("note" in df.columns) and (len(re.findall(f, n)) > 0)) or (len(re.findall(f, t)) > 0):
                feminine = True
        for m in mas:
            if masculine:
                break
            if (("note" in df.columns) and (len(re.findall(m, n)) > 0)) or (len(re.findall(m, t)) > 0):
                masculine = True

        if (feminine == True) and (masculine == False):
            genders += ["Feminine"]
        elif (feminine == False) and (masculine == True):
            genders += ["Masculine"]
        elif (feminine == True) and (masculine == True):
            genders += ["Multiple"]
        else:
            genders += ["Unclear"]
            
    df.insert(len(df.columns), "associated_genders", genders)
    return df

# Compare each manually annotated person name to all spaCy-labeled person names
def getAnnotFuzzyMatches(score_method, min_score):
    all_fuzzy_matches = []
    no_fuzzy_match = 0
    for n in unique_ppl:
        fuzzy_matches = process.extractBests(n, unique_persons, scorer=score_method, score_cutoff=min_score)
        if len(fuzzy_matches) == 0:
            no_fuzzy_match += 1
        else:
            all_fuzzy_matches = all_fuzzy_matches + fuzzy_matches
    return no_fuzzy_match, all_fuzzy_matches

# Compare each spaCy-labeled person name to all manually annotated person names
def getSpacyFuzzyMatches(score_method, min_score):
    all_fuzzy_matches = []
    no_fuzzy_match = 0
    for n in unique_persons:
        fuzzy_matches = process.extractBests(n, unique_ppl, scorer=score_method, score_cutoff=min_score)
        if len(fuzzy_matches) == 0:
            no_fuzzy_match += 1
        else:
            all_fuzzy_matches = all_fuzzy_matches + fuzzy_matches
    return no_fuzzy_match, all_fuzzy_matches

def getWordsSents(corpus_list):
    all_words, lower_words, all_sents = [], [], []
    for f in corpus_list.fileids():
        file_tokens = word_tokenize(corpus_list.raw(f))
        # Keep tokens that are alphabetical, numeric or a combination of letters and numbers (i.e., "4th")
        file_words = [t for t in file_tokens if t.isalpha() or re.match("\d+\w*",t)]
        all_words += [file_words]
        
        file_lower_words = [w.lower() for w in file_words]
        lower_words += [file_lower_words]
        
        file_sents = sent_tokenize(corpus_list.raw(f))
        all_sents += [file_sents]
    
    return all_words, lower_words, all_sents

def makeDescribeDf(field, desc_df):
    if field != "All":
        sub_df = (desc_df.loc[desc_df.field == field]).drop(columns=["eadid", "desc_id", "field", "description"])
    else:
        sub_df = desc_df.drop(columns=["eadid", "desc_id", "field", "description"])
    df_stats = sub_df.describe()
    df_stats = df_stats.drop(labels=["25%", "50%", "75%"], axis=0)  # remove the quartile calculations
    df_stats = df_stats.T
    df_stats.insert(len(df_stats.columns), "metadata_field", [field]*df_stats.shape[0])
    df_stats = df_stats.reset_index()
    df_stats = df_stats.rename(columns={"count":"total_descriptions", "index":"by"})
    df_stats = df_stats.set_index(["metadata_field", "by"])
    return df_stats