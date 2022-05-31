# gender-bias

Gender bias classification collaboration

Related repo: [annot-prep](https://github.com/thegoose20/annot-prep)

### To Do
* [Lucy] Define train/test split with sklearn
* [Seraphina] Make baseline classifiers
* [Lucy] Use ML classifier – possible resource constraints 
* [Seraphina] Pick a couple combination functions for doc representation

### Classification Overview
* Document classification - begin with TFIDF, experiment with 2ish combination functions for comparing document representations
* Multi-label - 9 labels from annotation taxonomy (see below)

### Inventory
**clf_data2**
* All `_docs.txt` files contain documents (individual metadata descriptions) separated by one newline, one pipe, and one newline: `"\n|\n"`
* All `_labels.txt` files contain comma-separated labels for each document, separated by one newline: `"\n"`
* The `training` files represent 20% of the data, randomly selected after shuffling, for training classification models
* The `validation` files represent 20% of the data, randomly selected after shuffling, for developing classification models
* The `blindtest` files represent 20% of the data, randomly selected after shuffling, for evaluating the final classification models

**SplitData** - Splitting the aggregated annotated dataset into training, validation, and blind test sets

### Data
* 55,000+ labels
* 15,419 sentences and 255,943 words (the first 20\% of the entire corpus of archival documentation)
* Descriptions from four metadata fields in the [Archives Online catalog](archives.collections.ed.ac.uk/) of the Centre for Research Collections at the University of Edinburgh
    1. Title - title of collection ("fonds"), subcollection, series, subseries, or item
    2. Scope and Contents - descriptions of the type of material (i.e. photos, journals, letters) 
    3. Biographical / Historical - descriptions of the people, places, and events associated with the archival items being described
    4. Processing Information - usually empty, but for the 30ish% of the time it's provided, names who wrote the description and the year they wrote it

### Annotation Taxonomy
Three categories of labels:
 1. Person Name: Unknown, Non-binary,* Feminine, Masculine
 2. Linguistic: Gendered Pronoun, Gendered Role, Generalization
 3. Contextual: Occupation, Omission, Stereotype, Empowering*

**Annotators did not find descriptions on which to apply these labels*
