import tensorflow as tf
from tensorflow import keras
from preprocessing import preprocessing
from constants import Config


class Classificator:
    def __init__(self, config):
        # load models
        self.sentiment_model = tf.keras.models.load_model(config.SENTIMENT_MODEL_PATH)
        self.topic_model = tf.keras.models.load_model(config.TOPIC_MODEL_PATH)
        self.topics = config.TOPICS
        
    
    def predict_classes(self, predictions):
        predicted_labels2 = []
        for (indx, probability) in enumerate(predictions[0]):
            if probability >= 0.3:
                predicted_labels2.append(indx)
        return [self.topics[label] for label in predicted_labels2]        


    def get_classification(self, review):
        try:
            preprocessed_review = preprocessing(review)
            sentiment = self.sentiment_model.predict([preprocessed_review])
            
            if sentiment[0] <= 0.5:
                sentiment = "negative"
                topic_predictions = self.topic_model.predict([preprocessed_review]) 
                predicted_topics = self.predict_classes(topic_predictions)
                negative_result = f"Sentiment: {[sentiment]}\nPredicted topic(s): {predicted_topics}"
                return negative_result
            else:
                sentiment = "positive"
                positive_result = f"Sentiment: {[sentiment]}"
                return positive_result
        
        except Exception as e:
            return f"Error processing the review: {str(e)}"



user_review = input("Enter a review: ")
classificator = Classificator(Config) 
result = classificator.get_classification(user_review)
print(result)