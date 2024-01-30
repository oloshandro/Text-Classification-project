# Text-Classification-project

## Description
This is taxi service reviews classification project. The aim is to create an efficient classification system to filter negative reviews, classify them according to the topics, and prioritize for the customer service fast response. 

## Structure of the repo:
* **datasets**

    Datasets and all data-related stuff is here
* **models**

    Saved models, suitable for testing 
* **notebooks**

    Place for jupyter notebooks. Notebooks could contain some analysis (dataset analysis, evalution results), demo, some ongoing work
* **src**

    Codebase for data preprocessing, training and testing models

## Requirements


## How to use
First of all, you will need to initialize your own .env file, create python virtual environment and install necessary packages:

```python3 -m venv venv
source venv/bin/activate
pip install -e . --no-deps
pip install -r requirements.txt
```

To train locally, on your own host machine, it's required to clone the repository with all the necessary data. Make sure you change ```WORK_DIR``` in `constants.py` to your local working directory.
Then run the following files:

```
python src/sentiment_model_train.py
python src/topic_model_train.py
```

To test the classification system on your own examples, run:
```
python src/classification_test
```

## Project status
The project is completed.