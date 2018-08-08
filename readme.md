# Sentiment Analysis of Twitter Data
# Final Year Project

## Overview

	This project applies machine learning techniques in order to classify tweets based on their emotional magnitude, every tweet will be classified as positive, neutral or negative.
	The tweets will be fetched in real-time using the Twitter API and classified using the Bernouli Naive Bayes classifier.

## Quick Start

	- Installing requirements using **pip**:

    Pip is a package manager for python, all the requirements needed in order for project to run are listed in **requirements.txt**,
	the following command must be run in order to install the requirements:
	
    - python -m pip install -r requirements.txt
	
	This should install all the requirements, alternatively, a full guide on how to install pip is found [here](https://pip.pypa.io/en/stable/installing/).

	- To run the project, run the live tweet classification script:
    - python main.py

    

## Tools and packages used

	In addition to python's standard library and built-in features, the following packages and tools were used for this project:

	- **[pandas](https://pandas.pydata.org/)** : Provides easy-to-use data structures and data analysis tools for the Python programming language.

	- **[nltk](https://www.nltk.org)**  : A natural language processing package for python that provides stemming and tokenization API for many languages including English.

	- **[sklearn](scikit-learn.org)** : Formally known as **scikit**, A machine learning package for python.

	- **[re](https://docs.python.org/3/library/re.html)**  : A regular expression library that is included in python's standard library.

	- **[dash](https://plot.ly/products/dash/)** : Built on top of Plotly.js, React, and Flask, Dash ties modern UI elements like dropdowns, sliders, and graphs to your analytical Python code.
	

## Training and Testing Datasets

	The training dataset is from [marrrcin's github repository](https://github.com/marrrcin/ml-twitter-sentiment-analysis), the dataset is in csv format and ready to be loaded using **pandas**, however some text cleaning was
	required before the tweets were ready for processing. The datasets are stored in the .\data folder.
	

## Contents
	
	Contents of the software are listed as follows:
	
	- Data folder which containts the datasets train.csv and the generated pol.csv
	- main.py
	- learn.py
	- twitter.py
	- settings.py
	- readme.md
	- requirements.txt

