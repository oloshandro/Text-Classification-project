import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocessing
from constants import Config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



class TopicClassificationModelBuild:
    def __init__(self, config):
        self.max_vocab_len = config.max_vocab_len
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size
        self.epochs_topic = config.epochs_topic
        self.num_topics = config.num_topics
        self.work_dir = config.WORK_DIR
        self.dataset_sentiment = config.DATASET_SENTIMENT
        self.vectorize_layer = layers.TextVectorization(
            max_tokens=config.max_vocab_len,
            output_mode='int',
            output_sequence_length=config.sequence_length
        )

    
    def preprocess_data(self, dataset_path):
        # read CSV into pandas dataframe
        reviews = pd.read_csv(dataset_path, encoding='utf-8')

        # preprocess text with methods defined in preprocessing.py
        data = [preprocessing(custom_comment) for custom_comment in reviews['custom_comment'].to_list()]
        data = np.array(data)

        topics = reviews.columns.values[3:].tolist()
        labels = reviews[topics].values
        labels = np.array(labels)

        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.vectorize_layer.adapt(X_train)

        return X_train, X_test, y_train, y_test
    


    def build_model(self):
        # lstm model with PreProcessing layer
        inputs = layers.Input(shape = (1,), dtype = 'string')
        x = self.vectorize_layer(inputs)
        x = layers.Embedding( 
            input_dim = self.max_vocab_len + 1, # int, the size of our vocabulary, maximum integer index + 1 
            output_dim = self.embedding_dim, # int, dimensions to which each words shall be mapped
            input_length = self.sequence_length, #Length of input sequences
            mask_zero=True #to ignore padding
            )(x)
        x = layers.LSTM(units=64, return_sequences=True)(x)
        x = layers.LSTM(units=32)(x)
        x = layers.Dense(units=32, activation = 'relu')(x)
        x = layers.Dropout(rate=0.25)(x)

        predictions = layers.Dense(units=self.num_topics, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, predictions, name = 'TOPIC_MODEL')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), #1e-3 = 0.001
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"])

        model.summary()
        return model
    

    def train_model(self, model, X_train, y_train, X_test, y_test, verbose=1):
        try:
            history = model.fit(X_train, y_train, 
                                batch_size=self.batch_size,
                                epochs=self.epochs_topic,
                                verbose=verbose)
            
            results = model.evaluate(X_test, y_test)
            print ('Test loss: {0}, Test accuracy: {1}'.format(results[0],results[1]))

            return history
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            return None
    

    
    def run_training(self, dataset_path):
        # Preprocess data
        X_train, X_test, y_train, y_test = self.preprocess_data(dataset_path)

        # Build a model
        model = self.build_model()

        # Train LSTM model
        history = self.train_model(model, X_train, y_train, X_test, y_test)

        # Return relevant information or results
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'lstm_model': model,
            'training_history': history
        }


# Initialize the model
model = TopicClassificationModelBuild(config=Config)

# Run the entire pipeline
model.run_training(dataset_path = Config.DATASET_TOPIC)