import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read the data from CSV file
data = pd.read_csv('ChatDataset.csv')

# Extract the chat texts and labels from the data
texts = data['Chat'].tolist()
labels = data['Label'].tolist()

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Classify a new text input
new_text = input("Enter a new text: ")
X_new = vectorizer.transform([new_text])
predicted_labels = model.predict(X_new)

# Get the top three predicted labels
top_labels = predicted_labels[0:3]

# Print the predicted labels
print("Predicted Tag:")
for label in top_labels:
    print(label)

# Ask for user feedback
feedback = input("Is the predicted label correct? (yes/no): ")

if feedback.lower() == "no":
    new_label = input("Please provide the correct label: ")
    
    # Add the new data to the dataset
    texts.append(new_text)
    labels.append(new_label)
    
    # Update the feature matrix and target vector
    X = vectorizer.transform(texts)
    y = np.array(labels)
    
    # Retrain the model
    model.fit(X, y)
    print("Model has been updated with user feedback.")
    
    # Update the data in the CSV file
    updated_data = pd.DataFrame({'Chat': texts, 'Label': labels})
    updated_data.to_csv('ChatDataset.csv', index=False)
    print("Chat Dataset has been updated.")
