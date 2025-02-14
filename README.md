# gender-bias

This repository contains text classification models, as well as the experiments undertaken to create them, to identify gendered and gender biased language.  The models were trained on a corpus of British English text extracted from an archival catalog's metadata descriptions, which consists of both historical and contemporary language (see [Data]((#Data))).

### Classification Model Overview
* Token Classifiers: multilabel task for Linguistic labels in the [taxonomy](#Annotation-Taxonomy), where words represented with word embeddings
* Sequence Classifiers: multiclass task for Person Name labels and Occupation label in the [taxonomy](#Annotation-Taxonomy), where words are represented with word embeddings
* Document Classifiers: multilabel task for Stereotype and Omission labels in the [taxonomy](#Annotation-Taxonomy), where each document is a description represented with TFIDF

### Table of Contents
* [Data](#Data)
* [Annotation Taxonomy](#Annotation-Taxonomy)
* [Overview of Directories](#Overview-of-Directories)
* [Related Resources](#Related-Resources)

## Data
* Descriptions from four metadata fields in the [University of Edinburgh Heritage Collections Archives catalog](archives.collections.ed.ac.uk/)
    1. **Title**: title of collection ("fonds"), subcollection, series, subseries, or item
    2. **Scope and Contents**: descriptions of the type of material (i.e. photos, journals, letters)
    3. **Biographical / Historical**: descriptions of the people, places, and events associated with the archival items being described
    4. **Processing Information**: usually empty, but for the ~30% of the collections it's provided and contains names who wrote the description and the year they wrote it
* 11,888 descriptions from over 1,000 archival collections (the first 20% of the Archives' catalog as of October 2020)
* 24,474 sentences and 399,957 words 
* 55,260 annotations (a.k.a. codes, labels) in the training dataset

## Annotation Taxonomy
Definitions and examples of each category and subcategory (the subcategories being the codes, or labels, applied during the manual annotation process) of the taxonomy listed below are available in [this paper](https://aclanthology.org/2022.gebnlp-1.4/).
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
**Annotators did not find descriptions on which to apply these labels according to the annotation instructions*

## Overview of Directories
* `document_classification` - experiments with document classifiers for targets as all labels, Person Name labels, and Stereotype and Omission labels
* `token_classification` - experiments with multilabel token classifiers, multiclass sequence classifiers, and cascades of classifiers (meaning sequential combinations of token, sequence, and document classifiers); the experiments in this directory correspond to the cascades in the associated paper (see next section), *i.e.*, Experiment1 = Cascade 1
* `word_embeddings` - evaluating relevance of SpaCy's sense2vec (contextual word embeddings) and of GloVe embeddings for the classification task, and training custom fastText embeddings

## Associated Paper
```
@inproceedings{Havens_Bach_Terras_Alex_2025, 
  author={Havens, Lucy and Bach, Benjamin and Terras, Melissa and Alex, Beatrice},
  title={{Investigating the Capabilities and Limitations of Machine Learning for Identifying Bias in English Language Data with Information and Heritage Professionals}}, 
  booktitle={CHI ’25: Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems}, 
  publisher={ACM},
  address={New York}
  location={Yokohama, Japan},
  DOI={https://doi.org/10.1145/3706598.3713217}, 
  year={2025}, 
  pages={22} 
}
```

## Related Resources
* GitHub repos: 
  * [annot-prep](https://github.com/thegoose20/annot-prep)
  * [annot](https://github.com/thegoose20/annot)
* Observable Notebooks: 
  * [Confusion Matrices of Annotated Archival Metadata Descriptions](https://observablehq.com/@thegoose20/confusion-matrices)
  * [Exploratory Analysis of Archival Metadata](https://observablehq.com/d/0091bad1ddecc57f)
  * [Exploratory Analysis of Annotated Data](https://observablehq.com/d/b61080669b52aa93)
* Publications:
  * On the research methodology: [Situated Data, Situated Systems (Havens et al., 2020)](https://aclanthology.org/2020.gebnlp-1.10.pdf)
  * On the coding taxonomy and training data: [Uncertainty and Inclusivity in Gender Bias Annotation (Havens et al., 2022)](https://aclanthology.org/2022.gebnlp-1.4v2.pdf)
  * On the classification experiments and model performance analysis: [Recalibrating Machine Learning for Social Biases (Havens, 2024)](https://era.ed.ac.uk/handle/1842/41420)
* Datasets:
  * [Annotated datasets](https://doi.org/10.7488/ds/7540)
  * [Text classification models' input data](https://doi.org/10.7488/ds/7539)