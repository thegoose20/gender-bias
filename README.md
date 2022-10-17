# gender-bias

Gender bias classification collaboration

### To Do
- [x] Define train/test split with sklearn
- [X] Make baseline classifiers
- [ ] Make ML classifier with feature engineering – possible resource constraints
- [ ] Use DL classifier
- [ ] [Seraphina] Pick a couple combination functions for doc representation

### Classification Overview
* Document classification - begin with TFIDF, experiment with 2ish combination functions for comparing document representations
* Multi-label - 9 labels from annotation taxonomy (see below)

### Table of Contents
* [Data](#Data)
* [Annotation Taxonomy](#Annotation-Taxonomy)
* [Index of Files](#Index-of-Files)
* [Related Repos](#Related-Repos)

## Data
* 55,000+ labels
* 15,419 sentences and 255,943 words (the first 20\% of the entire corpus of archival documentation)
* Descriptions from four metadata fields in the [Archives Online catalog](archives.collections.ed.ac.uk/) of the Centre for Research Collections at the University of Edinburgh
    1. **Title**: title of collection ("fonds"), subcollection, series, subseries, or item
    2. **Scope and Contents**: descriptions of the type of material (i.e. photos, journals, letters)
    3. **Biographical / Historical**: descriptions of the people, places, and events associated with the archival items being described
    4. **Processing Information**: usually empty, but for the ~30% of the collections it's provided and contains names who wrote the description and the year they wrote it

## Annotation Taxonomy
Definitions and examples of each category and subcategory of the taxonomy listed below are available in [this paper](https://aclanthology.org/2022.gebnlp-1.4/).
```
Gendered and Gender Biased Language
├── Person Name
│   ├── Unknown
│   ├── Non-binary*
│   ├── Feminine
│   └── Masculine
├── Linguistic
│   ├── Generalization
│   ├── Gendered Pronoun
│   └── Gendered Role
└── Contextual
    ├── Empowering*
    ├── Occupation
    ├── Omission
    └── Stereotype
```
**Annotators did not find descriptions on which to apply these labels*

## Index of Files
* `clf_data/`
  * The `_docs.txt` files contain documents (individual metadata descriptions) separated by one newline, one pipe, and one newline: `"\n|\n"`
  * The `_labels.txt` files contain comma-separated labels for each document, separated by one newline: `"\n"`
  * The `train_` files contain 60% of the data *from each type of metadata field*, randomly selected (`random_state=7`), for training classification models
  * The `validate_` files contain 20% of the data *from each type of metadata field*, randomly selected (`random_state=7`), for developing classification models
  * The `blindtest_` files contain 20% of the data *from each type of metadata field*, randomly selected (`random_state=7`), for evaluating the final classification models

* `clf_data2/`
  * The `_docs.txt` files contain documents (individual metadata descriptions) separated by one newline, one pipe, and one newline: `"\n|\n"`
  * The `_labels.txt` files contain comma-separated labels for each document, separated by one newline: `"\n"`
  * The `training_` files contain 60% of the data, randomly selected after shuffling, for training classification models
  * The `validation_` files contain 20% of the data, randomly selected after shuffling, for developing classification models
  * The `blindtest_` files contain 20% of the data, randomly selected after shuffling, for evaluating the final classification models

* `SplitData.ipynb` - Splitting the aggregated annotated dataset into training, validation, and blind test sets
* `DocumentClassifiers.ipynb` - Baseline document classification models using Multinomial Naive Bayes, Logistic Regression, and Random Forest algorithms

## Related Resources
* GitHub repos: 
  * [annot-prep](https://github.com/thegoose20/annot-prep)
  * [annot](https://github.com/thegoose20/annot)
* Observable Notebooks: 
  * [Confusion Matrices of Annotated Archival Metadata Descriptions](https://observablehq.com/@thegoose20/confusion-matrices)
  * [Exploratory Analysis of Archival Metadata](https://observablehq.com/d/0091bad1ddecc57f)
  * [Exploratory Analysis of Annotated Data](https://observablehq.com/d/b61080669b52aa93)
