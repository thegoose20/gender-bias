# gender-bias

Gender bias classification models

### Classification Overview
* Token Classifiers: multilabel task for Linguistic labels in Taxonomy (see below), words represented with word embeddings
* Sequence Classifiers: multiclass task for Person Name labels and Occupation label in Taxonomy, words represented with word embeddings
* Document Classifiers: multilabel task for Stereotype and Omission labels in Taxonomy, each document is a description represented with TFIDF

### Table of Contents
* [Data](#Data)
* [Annotation Taxonomy](#Annotation-Taxonomy)
* [Overview of Directories](#Overview-of-Directories)
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

## Overview of Directories
* `analysis` - analysis of model predictions (outputs)
* `document_classification` - experiments with document classifiers for targets as all labels, Person Name labels, and Stereotype and Omission labels
* `statistical_significance` - paired bootstrap test on models (stat. sign. testing isn't really suited to NLP tasks, though)
* `token_classification` - experiments with multilabel token classifiers, multiclass sequence classifiers, and cascades of classifiers (token+sequence+document)
* `word_embeddings` - evaluating relevance of SpaCy's sense2vec (contextual word embeddings) and of GloVe embeddings for the classification task, and training custom fastText embeddings

## Related Resources
* GitHub repos: 
  * [annot-prep](https://github.com/thegoose20/annot-prep)
  * [annot](https://github.com/thegoose20/annot)
* Observable Notebooks: 
  * [Confusion Matrices of Annotated Archival Metadata Descriptions](https://observablehq.com/@thegoose20/confusion-matrices)
  * [Exploratory Analysis of Archival Metadata](https://observablehq.com/d/0091bad1ddecc57f)
  * [Exploratory Analysis of Annotated Data](https://observablehq.com/d/b61080669b52aa93)
  * [Document Classification Error Analysis](https://observablehq.com/@thegoose20/lr-doc-classification-error-analysis)
