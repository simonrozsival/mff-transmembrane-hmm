# Transmembrane Protein Topology Prediction Using Hidden Markov Models

This project was created for the course Machine Learning in Bioinformatics at the Faculty of Mathematics and Physics of the Charles University in Prague by Šimon Rozsíval.

## Requirements

- Python 3.5
- Natural Language Toolkit - http://www.nltk.org/
- scikit-learn 18

## HMM library

The `src/hmm.py` script was downloaded from http://www.mit.edu/course/6/6.863/OldFiles/python/old/nltk-contrib-1.4.2/build/lib/nltk_contrib/unimelb/tacohn/hmm.py and edited to work with Python 3 and NLTK (instead of NLTK-lite) and also some parts of the scripts (demo) were removed.

## Training data

The data used for training are in the `set160.labels.txt` file. This file contains 160 proteins with correct anotations and was taken from http://www.cbs.dtu.dk/~krogh/TMHMM/.

## Run the script

Just execute using `python3 train.py`. This script outputs all results and statistics on the standard output. The trained model is not serialized and saved anywhere.

## Results

The `results` directory contains some results of experiments on the test data using different models.