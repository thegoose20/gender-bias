import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import re
from thefuzz import fuzz, process


# ---------------------------------------------------------------------------------------------
# Analysis_GenderedPronounsRoles
# ---------------------------------------------------------------------------------------------
def addMatch(ismatch, matches):
    if ismatch != None:
        matched_word = ismatch[0]
        if matched_word in matches.keys():
            matches[matched_word] += 1
        else:
            matches[matched_word] = 1
    return matches

def checkForMatch(text, matches, patterns):
    for pattern in patterns:
        ismatch = pattern.match(text)
        matches = addMatch(ismatch, matches) 
    return matches

def countMatches(descs,patterns):
    matches = dict()
    for desc in descs:
        if type(desc) == list:
            for word in desc:
                if len(word) > 1:
                    word_uncapitalized = word[0].lower() + word[1:]
                else:
                    word_uncapitalized = word.lower()
                matches = checkForMatch(word_uncapitalized, matches, patterns)                
        elif type(desc) == str:
            desc = desc.lower()
            matches = checkForMatch(desc, matches, patterns)
        else:
            raise ValueError
    return matches 


# ---------------------------------------------------------------------------------------------
# Analysis_NamedPeople
# ---------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------
# Analysis_CommonlyAnnotatedText
# ---------------------------------------------------------------------------------------------
def getFieldRatios(df_field_values):
    total = sum(df_field_values)
    ratios = []
    for v in df_field_values:
        ratios += [v/total]
    return ratios

def getValueCountsDataFrame(label_series, label_name):
    df = pd.DataFrame(label_series)
    df = df.reset_index()
    df = df.rename(columns={"index":"text", "text":"occurrence"})
    df.insert(len(df.columns), "label", [label_name]*df.shape[0])
    return df

# ---------------------------------------------------------------------------------------------
# Analysis_LengthsAndOffsets
# ---------------------------------------------------------------------------------------------
# Write each string to a txt file named with the string's ID
def strToTxt(ids, strs, filename_prefix, dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    zero_padding = len(str(ids[-1]))
    i, maxI = 0, len(ids)
    while i < maxI:
        d_id = str(ids[i])
        padding = zero_padding - len(d_id)  # pad with zeros so file order aligns with DataFrame order
        id_str = ("0"*padding) + d_id
        filename = filename_prefix+id_str+".txt"
        f = open((dir_path+filename), "w", encoding="utf8")
        f.write(strs[i])
        f.close()
        i += 1
    return "Files written to "+dir_path+"!"


# INPUT: list of strings, list of ids for those strings, list of start offsets for those strings, 
#        list of end offsets for those strings (offsets in the brat rapid annotation tool's standoff format)
# OUTPUT: two dictionaries, both with ids as keys, and one with lists of tokens as values and the other 
#         with lists of those tokens' offsets as values
def getTokensAndOffsetsFromStrings(list_of_strings, list_of_ids, list_of_start_offsets, list_of_end_offsets):
    tokens_dict = dict.fromkeys(list_of_ids)
    offsets_dict = dict.fromkeys(list_of_ids)
    j, maxJ = 0, len(list_of_strings)
    
    while j < maxJ:

        # Get the string's ID
        s_id = list_of_ids[j]
        
        # Get the start and end offsets of the description
        s_start_offset = list_of_start_offsets[j]
        
        # Get the description string and its tokens
        s = list_of_strings[j]
        s_tokens = word_tokenize(s)
        
        # Get the start and end offsets of the first token
        first_t = s_tokens[0]
        t_start_offset = s_start_offset
        t_end_offset = s_start_offset + len(first_t)
        tokens_dict[s_id] = [first_t]
        offsets_dict[s_id] = [tuple((t_start_offset, t_end_offset))]
        prev_positions = len(first_t)
        sub_s = s[(t_end_offset-s_start_offset):]
        sub_tokens = word_tokenize(sub_s)
        
        # Get the start and end offsets of the remaining tokens
        for t in sub_tokens:
            i = sub_s.index(t)
            t_start_offset = i + s_start_offset + prev_positions
            t_end_offset = t_start_offset + len(t)
            tokens_dict[s_id] += [t]
            offsets_dict[s_id] += [tuple((t_start_offset, t_end_offset))]
            sub_s = sub_s[i+len(t):]
            prev_positions += len(t)+i
            
        assert t_end_offset <= list_of_end_offsets[j], "The last token's end offset should not be beyond the string's end offset"
            
        j += 1
    
    return tokens_dict, offsets_dict