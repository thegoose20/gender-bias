import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import re, os
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
        sub_df = (desc_df.loc[desc_df.field == field]).drop(columns=["description_id", "field", "description", "start_offset", "end_offset"])
    else:
        sub_df = desc_df.drop(columns=["description_id", "field", "description", "start_offset", "end_offset"])
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
# INPUT: file path to a document of metadata descriptions (str) and a sorted (alphabetically, A-Z) list of file names
# OUTPUT: a dictionary of metadata description ids and the associated 
#         description text, field name and offsets contained in the input file
fields = ["Identifier", "Title", "Scope and Contents", "Biographical / Historical", "Processing Information"]

def updateDescDict(sub_f_string, d, f, desc_dict, did, last_end_offset):
    start_offset = sub_f_string.index(d) + last_end_offset
    end_offset = start_offset + len(d) + 1
    
    desc_dict[did] = dict.fromkeys(["description", "file", "start_offset", "end_offset"])
    
    desc_dict[did]["description"] = d 
    desc_dict[did]["file"] = f
    desc_dict[did]["start_offset"] = start_offset
    desc_dict[did]["end_offset"] = end_offset

    did += 1
    
    return desc_dict, did, end_offset

def getDescriptionsInFiles(dirpath, file_list):
    desc_dict = dict()
    did = 0
    for f in file_list:
        # Get a string of the input file's text (metadata descriptions)
        f_string = open(os.path.join(dirpath,f),'r').read()
        descs = f_string.split("\n\n")
        last_end_offset = 0
        for d in descs:
            sub_f_string = f_string[last_end_offset:]
            if d != "":
                desc_dict, did, last_end_offset = updateDescDict(sub_f_string, d, f, desc_dict, did, last_end_offset)                
    return desc_dict


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
        try:
            s_tokens = word_tokenize(s)
        except TypeError:
            print(s)
        
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
            old_t = t
            if (t == "''") or (t == "``"):
                t_A = '"'
                t_B = "''"
                if (t_A in sub_s) and (t_B in sub_s):
                    i_A = sub_s.index(t_A)
                    i_B = sub_s.index(t_B)
                    if i_A < i_B:
                        t = t_A
                    else:
                        t = t_B
                elif (t_A in sub_s) and (not t_B in sub_s):
                    t = t_A
                elif (not t_A in sub_s) and (t_B in sub_s):
                    t = t_B
                else:
                    print("t:",t, "\n sub_s:",sub_s)
            try:
                i = sub_s.index(t)
            except ValueError:
                print("old_t:",t)
                
            t_start_offset = i + s_start_offset + prev_positions
            t_end_offset = t_start_offset + len(t)
    
            assert t_end_offset <= list_of_end_offsets[j], "The last token's ({}) end offset ({}) should not be beyond the string's end offset ({}):{}, {}".format(t, t_end_offset, list_of_end_offsets[j], s_id, s)
            
            tokens_dict[s_id] += [t]
            offsets_dict[s_id] += [tuple((t_start_offset, t_end_offset))]
            sub_s = sub_s[i+len(t):]
            prev_positions += len(t)+i
            
                        
        j += 1
    
    return tokens_dict, offsets_dict


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


def turnStrTuplesToIntTuples(list_of_tuples):
    new_list = []
    for t in list_of_tuples:
        if ", " in t:
            t = t[1:-1].split(", ")
        elif "," in t:
            t = t[1:-1].split(",")
        new_t = tuple((int(t[0]), int(t[1])))
        new_list += [new_t]
    return new_list
