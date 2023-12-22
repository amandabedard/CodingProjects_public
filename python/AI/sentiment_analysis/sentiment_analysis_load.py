import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from joblib import load

# Make sure to download the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the preprocess_text function exactly as it was during training
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word not in stopwords.words('english')]

# Load the trained model
model = load('sentiment_analysis_model.joblib')

# Function to predict sentiment of a new sentence
def predict_sentiment(sentence):
    return model.predict([sentence])[0]

# Get user input
user_sentence = input("Enter a sentence to analyze its sentiment: ")

# Predict and display the sentiment
predicted_sentiment = predict_sentiment(user_sentence)
print(f"The predicted sentiment of the sentence is: {predicted_sentiment}")