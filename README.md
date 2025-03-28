# gender-bias

This repository contains text classification models, as well as the experiments undertaken to create them, to identify gendered and gender biased language.  The models were trained on a corpus of British English text extracted from an archival catalog's metadata descriptions, which consists of both historical and contemporary language (see [Data]((#Data))).

## Table of Contents
1. [Classification Model Overview](#Classification-Model-Overview)
2. [Data](#Data)
3. [Model Classes and Labels](#Model-Classes-and-Labels)
4. [Repo Overview](#Repo-Overview)
5. [Setup](#setup)
6. [Associated Paper](#Associated-Paper)
7. [Related Resources](#Related-Resources)

## 1. Classification Model Overview
* Token Classifiers: multilabel task for Linguistic labels in the [Taxonomy of Gendered and Gender Biased Language](#Model-Classes-and-Labels), where words are represented with custom word embeddings
* Sequence Classifiers: multiclass task for Person Name labels and Occupation label in the [Taxonomy of Gendered and Gender Biased Language](#Model-Classes-and-Labels), where words are represented with custom word embeddings
* Document Classifiers: multilabel task for Stereotype and Omission labels in the [Taxonomy of Gendered and Gender Biased Language](#Model-Classes-and-Labels), where each document is a description represented with TFIDF

## 2. Data
Training data for all classification models can be downloaded from the [University of Edinburgh's DataShare platform](https://doi.org/10.7488/ds/7539).  The models' training data was created from an aggregated, annotated dataset of descriptions from four metadata fields in the [University of Edinburgh Heritage Collections' Archives catalog](https://archives.collections.ed.ac.uk):
    1. **Title**: title of collection ("fonds"), subcollection, series, subseries, or item
    2. **Scope and Contents**: descriptions of the type of material (i.e. photos, journals, letters)
    3. **Biographical / Historical**: descriptions of the people, places, and events associated with the archival items being described
    4. **Processing Information**: usually empty, but for the ~30% of the collections it's provided and contains names who wrote the description and the year they wrote it
This dataset can also be downloaded from [DataShare](https://doi.org/10.7488/ds/7540) and consists of:
* 11,888 descriptions from over 1,000 archival collections (the first 20% of the Archives' catalog as of October 2020)
* 24,474 sentences and 399,957 words 
* 55,260 annotations (a.k.a. codes, labels) in the training dataset

## 3. Model Classes and Labels
Models were trained to classify text with the subcategories, or labels, of the *Taxonomy of Gendered and Gender Biased Language* in a multiclass or multilabel classification task.  Definitions and examples of the Taxonomy's categories and subcategories, which are listed below, are available in [this paper](https://aclanthology.org/2022.gebnlp-1.4/).
```
Taxonomy of Gendered and Gender Biased Language
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
**Note that annotators did not find descriptions on which to apply these labels according to the annotation instructions.*

## 4. Repo Overview
* `experiments/` - documentation of experiments undertaken to create text classification models to label catalog metadata descriptions according to the [Taxonomy of Gendered and Gender Biased Language](#Model-Classes-and-Labels)
  * `analysis/` - analysis of manually annotated and classified data
  * `document_classification/` - experiments with document classifiers where the models classify descriptions using all the Taxonomy's labels, only Person Name labels, and only Stereotype and Omission labels
  * `token_classification/` - experiments with multilabel token classifiers, multiclass sequence classifiers, and cascades of classifiers (meaning sequential combinations of token, sequence, and document classifiers); the experiments in this directory correspond to the cascades in the [associated paper](#associated-paper) (*i.e.*, `Experiment1` = "Cascade 1")
  * `word_embeddings/` - evaluating relevance of SpaCy's sense2vec (contextual word embeddings) and of GloVe embeddings for the classification task, and training custom fastText embeddings
* `final_models/` - text classification models available for reuse and guidance for reusing them (see `models/README.md` for detail)
* `environment.yml` - file for creating a virtual environment, `gender-bias`, using conda
* `mac-environment.yml` - file for creating a virtual environment on a Mac OS, `gender-bias-env`, using conda
* `requirements.txt` - file for creating a virtual environment using pip


## 5. Setup
We recommend using [conda](#https://conda.org) to reuse the code and resources in this repo (though we also provide a `requirements.txt` file if you prefer to use pip).  If you don't have conda, follow [these instructions](#https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to download and install Anaconda or Miniconda.

### 5.1 To re-use the final models

**Step 1:** Clone the repository from your command line.  

*If you're not familiar with command line tools, check out [this Bash tutorial](https://programminghistorian.org/en/lessons/intro-to-bash) (for Mac and Linux) or [this PowerShell tutorial](https://programminghistorian.org/en/lessons/intro-to-powershell) (for Windows) from the Programming Historian.*
```
git clone https://github.com/thegoose20/gender-bias.git
```

**Step 2:** Enter the directory (folder) you've just cloned.
```
cd gender-bias
```

**Step 3:** Setup a virtual environment using conda with one of the repo's YAML files.
```
conda env create -f environment.yml
```

**Step 4:** Activate your newly created virtual environment.
```
conda activate gender-bias
```

**Step 5:** Initialize the git repository.  

*If you're not familiar with git or GitHub, checkout GitHub's [Quick Start](https://docs.github.com/en/get-started/start-your-journey) and [Using Git](https://docs.github.com/en/get-started/using-git) documentation.*
```
git init
```

**Step 6:** Download the classification model's training data from [Edinburgh DataShare](#https://doi.org/10.7488/ds/7539).

**Step 7:** Create a directory in the gender-bias repo called `data`.
```
mkdir data
```

**Step 8:** Create a sub-directory called `token_clf_data`.
```
cd data
mkdir token_clf_data
```

**Step 9:** Create a sub-sub-directory called `model_input`.
```
cd token_clf_data
mkdir model_input
```

**Step 10:** Move the downloaded `token_train.csv` and `token_validate.csv` files under the newly created `data/token_clf_data/model_input` directory.

**Step 11:** Create Plain Text (TXT) files of the latest catalog metadata from the University of Edinburgh's Archives using [OAI-PMH](https://www.openarchives.org/pmh/) using the code in this repo.   `descriptions_by_fonds/` DATA FILES IN `word_embeddings/WordEmbeddings.ipynb` CODE B/C NOT AVAILABLE ONLINE!  BY CREATING FILES (OR WORD EMBEDDINGS?) FROM TWO FILES IN STEP 10?  IGNORING PIPES SEPARATING EACH DESCRIPTION???

Now you're ready to start running the text classifiers in the `final_models` directory!  

When you're done, shut down your virtual environment by entering the following in the command line:
```
conda deactivate
```
Re-activate the environment by running the command in step 4.

### 5.2 To re-run the experiments

If you'd like to run the code in the `experiments` directory, you'll need use additional files in the downloaded data and organize them into directories according to the `experiments/token_classification/config.py` file.

**Step 1:** Create a directory in your gender-bias repo called `data`.
```
mkdir data
```

**Step 2:** Create a sub-directory called `token_clf_data`.
```
cd data
mkdir token_clf_data
```

**Step 3:** Create a sub-sub-directory called `experiment_input`.
```
cd token_clf_data
mkdir experiment_input
```

**Step 4:** Create a sub-directory called `doc_clf_data`.
```
cd ..
mkdir doc_clf_data
```

**Step 5:** Create a sub-sub-directory called `model_input`.
```
cd doc_clf_data
mkdir model_input
```

**Step 6:** Move the downloaded `token_5fold.csv` and `document_5fold.csv` files under `data/token_clf_data/experiment_input/`.

**Step 7:** Move the `train_docs.txt`, `train_labels.txt`, `validate_docs.txt`, `validate_labels.txt`, `blindtest_docs.txt`, and `blindtest_labels.txt` files under `data/doc_clf_data/model_input/`.

The remaining downloaded data files are informative and not needed to run the code in the `experiments` directory.

## 6. Associated Paper
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

## 7. Related Resources
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
  * [Text classification models' training data](https://doi.org/10.7488/ds/7539)