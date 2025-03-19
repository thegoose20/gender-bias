# Index
An index of trained models available for running gender biased text classification.  All models were exported with [joblib](https://joblib.readthedocs.io) and can be loaded with:
```
joblib.load(<FILEPATH>)
```
where `<FILEPATH>` is replaced with a string containing the file path to the model you wish to run, for example:
```
mlb = joblib.load("mlb_linglabels.joblib")
trained_clf = joblib.load("cc-rf_F-fasttext100_T-linglabels.joblib")
y = trained_clf.predict(X)
predicted_labels = mlb.inverse_transform(y)
```
where `X` is a feature matrix (preprocessed text), `y` is a binarized representation of the classifier's predictions (e.g., `[0, 0, 1], [1, 0, 0], [0, 0, 0], ...`), and `predicted_labels` has the classifier's predictions represented as text (e.g., `[['Generalization'], ['Gendered-Pronoun'], [], ...`).

## embeddings/
This directory contains word embedding models for creating features to input into multilabel token classifiers.
* `custom_fasttext/fasttext_cbow_100d.model`: a custom word embedding model, specifically 100-dimension FastText embeddings trained with Continous Bag-of-Words (CBOW) architecture on metadata descriptions from the Heritage Collections' Archives' catalogue found in directory `data/descriptions_by_fonds/`, where all tokens are lowercased


## multilabel_document/
This directory contains scikit-learn-based models for running trained multilabel document classifiers to identify gender biases in text.
* `count_vectorizer.joblib`: a scikit-learn [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) fit on a training subset of metadata descriptions from the Heritage Collections' Archives' catalog
* `mlb_so.joblib`: a scikit-learn [MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) fit on the **Contextual** category's *Stereotype* and *Omission* labels, where the presence of a label is indicated with a 1 and the absence of a label is indicated with a zero (for example, a description with a *Stereotype* label and without an *Omission* label would be represented as `[0, 1]`)
* `sgd-svm_F-tfidf-ling_T-so.joblib`: a scikit-learn [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) using a Support Vector Machines (SVM) loss function (`loss='hinge'`) in a [one-vs.-rest setup](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) for multilabel document classification using binary relevance, trained on the above-mentioned training subset of metadata descriptions from the Heritage Collections' Archives' catalog
* `tfidf_transformer.joblib`: a scikit-learn [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) fit on the vectorized text of the above-mentioned training subset of metadata descriptions from the Heritage Collections' Archives' catalog, used to represent text to input into the SGDClassifier as a term frequency-inverse document frequency (TFIDF) matrix 

## multilabel_token/
This directory contains cikit-learn-based models for running trained multilabel token classifiers to identify gender biases in text.
* `cc-rf_F-fasttext100_T-linglabels.joblib`: a scikit-multilearn [Classifier Chain](http://scikit.ml/api/skmultilearn.problem_transform.cc.html) of scikit-learn [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) trained to perform multilabel token classification on text represented with the 100-dimension custom FastText embeddings (see `custom_fasttext/fasttext_cbow_100d.model` above), assigning the **Linguistic** category of labels (*Gendered-Pronoun*, *Gendered-Role*, *Generalization*) to input text
* `mlb_linglabels.joblib`: a scikit-learn [MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) fit on the **Linguistic** category's *Gendered-Pronoun*, *Gendered-Role*, and *Generalization* labels, where the presence of a label is indicated with a 1 and the absence of a label is indicated with a zero (for example, a token with only a *Gendered-Role* label would be represented as `[0, 1, 0]`)