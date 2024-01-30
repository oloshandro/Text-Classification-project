class Config:
    ## ----------------------File and Folders------------------------------------Start
    WORK_DIR = 'D:/DEV2/Text-Classification-project/'
    DATASET_SENTIMENT = WORK_DIR + '/datasets/annotated_data_sentiment.csv'
    DATASET_TOPIC = WORK_DIR + '/datasets/annotated_data_topic_classification.csv'
    SRC_PATH = WORK_DIR + '/src'
    SENTIMENT_MODEL_PATH = WORK_DIR + '/models/sentiment_model'
    TOPIC_MODEL_PATH = WORK_DIR + '/models/topic_model'


    ## ----------------------Models------------------------------------Start
    # constants
    embedding_dim = 50
    sequence_length = 100 
    max_vocab_len = 10000
    batch_size = 64
    epochs_sentiment = 6
    epochs_topic = 10
    num_topics = 10
    TOPICS = ['1 Pricing and Fairness', '2 Driver professionalism', '3 Driver behaviour', '4 Customer Service', '5 Application', '6 Lost things', '7 Vehicle Condition', '8 Safety & reliability', '9 General bad', '10 Other']
  