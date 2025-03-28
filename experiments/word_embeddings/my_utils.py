import pandas as pd
import numpy as np
import re

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