<p align="center">
  <img src="https://i.imgur.com/SPYT1zV.png" width="154">
  <h1 align="center">Discriminative and generative humor models</h1>
  <p align="center">Two pipelines that implement <b>text preprocessing, augmenting, model training, and model scoring or text generation</b> of data
  received from <b>Reddit</b>.
Implemented in Python.</p>
  <p align="center">
	<a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/built%20with-Python3-C45AEC.svg" />
    </a>
    <a href="https://matplotlib.org">
	<img src="https://img.shields.io/badge/bulid with-Matplotlib-7fffd4.svg">
    </a>
    <a href="https://seaborn.pydata.org/">
	<img src="https://img.shields.io/badge/bulid with-Seaborn-F70D1A.svg">
    </a>
    <a href="https://pytorch.org/">
	<img src="https://img.shields.io/badge/bulid with-PyTorch-DFFF00.svg">
    </a>
    <a href="https://scikit-learn.org/">
	<img src="https://img.shields.io/badge/bulid with-Sklearn-FD349C.svg">
    </a>
    <a href="https://numpy.org/doc/stable/index.html">
	<img src="https://img.shields.io/badge/bulid with-NumPy-1589FF.svg">
    </a>
    <a href="https://pandas.pydata.org/">
	<img src="https://img.shields.io/badge/bulid with-Pandas-FFFF00.svg">
    </a>
    <a href="https://scipy.org/">
	<img src="https://img.shields.io/badge/bulid with-SciPy-CCCCFF.svg">
    </a>
  </p>


## Table of contents
- [Project Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)
  
### **Structure**
* **checkpoints/** --> pre-trained model weights
* **data/**
    - jokes_dataset.csv --> raw_data
    - aug_jokes_***.csv --> data after preprocessing and augmentation (for experiments)
    - jokes_***.csv --> data after preprocessing (for experiments)
* **notebooks/**
    * **demo/**
        * **NLP_humor_EDA.ipynb** --> exploratory data analysis of dataset
        * **multiclassification_pipeline.ipynb** --> demo of usage multiclassification pipeline
        * **jokes_generation_pipeline(CPU_run).ipynb** --> demo of usage jokes generation pipeline (was running on CPU)
        * **jokes_generation(GPU_run).ipynb** --> notebook contains almost the entire pipeline code for generating jokes, but without dividing into modules(classes) and preprocessing. It was used in Google Colab to train the model on the GPU.
    * **experiments/**
        * **classification/**
        * **generation/**
        * **preproc/**
        * **regression/**
* **src/**
    * **augmenting/** --> class for text augmentation
    * **evaluate/** --> metrics for classification models
    * **loading_dataset/** --> dataset loader for generation model
    * **models_full_joke/** --> custom models for jokes that consist of 'title' + 'body'
    * **pipeline/** --> pipelines to unite data processing, model training and predictions and counting scores or text generation for it's performance
    * **preprocessing/** --> class for data preprocessing
    * **text2seq/** --> class for text conversion into numbers
    - **config.py**
    - **constants.py** 
    - **core.py** --> base class transformer
    - **schema.py** --> contains all columns
    - **train_test_split.py** --> class wrapper of sklearn train_test_split
    - **utils.py** --> additional utils for different classes
    
### **Installation**  
__Important:__ depending on your system, make sure to use `pip3` and `python3` instead.  
**Environment**   
* Python version 3.9  
* All dependencies are mentioned in *requirements.txt*

### **Usage**
This repository contains 2 pipelines:
- jokes classification
- jokes generation

To run one of pipelines, see the usage **[classification demo](https://github.com/DoktaPola/Socials/blob/master/main.py)**
or **[generation demo](https://github.com/DoktaPola/Socials/blob/master/main.py)**.

* First of all, all needed for pipeline instances have to be called and passed to the appropriate pipeline. 
* Then, prepare_data() has to be called, in order to process the dataset before training.
* Next, call train_model().
* And finally, predict() or generate().

---

> **Disclaimer**<a name="disclaimer" />: Please Note that this is a research project. I am by no means responsible for any usage of this tool. Use on your own behalf. I'm also not responsible if your accounts get banned due to extensive use of this tool.
