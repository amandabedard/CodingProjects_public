from joblib import dump
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Read data from CSV file
df = pd.read_csv('sentiment_dataset.csv')
sentences = df['Sentence'].values
labels = df['Sentiment'].values

# Preprocessing: Tokenization and Stopwords removal
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word not in stopwords.words('english')]

# Vectorization and Model Pipeline
model = make_pipeline(
    CountVectorizer(analyzer=preprocess_text),
    MultinomialNB()
)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Testing the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Function to predict sentiment of a new sentence
def predict_sentiment(sentence):
    return model.predict([sentence])[0]

# Get user input
user_sentence = input("Enter a sentence to analyze its sentiment: ")

# Predict and display the sentiment
predicted_sentiment = predict_sentiment(user_sentence)
print(f"The predicted sentiment of the sentence is: {predicted_sentiment}")

# Saving the model
dump(model, 'sentiment_analysis_model.joblib')
print("Model saved successfully!")