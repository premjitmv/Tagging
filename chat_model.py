import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read the data from CSV file and preprocess it
data = pd.read_csv('ChatDataset.csv')
texts = data['Chat'].tolist()
labels = data['Label'].tolist()

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'chat_model.joblib')
print("Model has been saved as 'chat_model.joblib'.")

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.joblib')
print("Vectorizer has been saved as 'vectorizer.joblib'.")
