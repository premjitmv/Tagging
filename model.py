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
predicted_proba = model.predict_proba(X_new)[0]

# Get the top predicted labels and their corresponding probabilities
top_labels = model.classes_[np.argsort(predicted_proba)][::-1][:5]
top_probabilities = np.sort(predicted_proba)[::-1][:5]

# Create a list of predicted tags and their probabilities (in percentage)
predicted_tags = [(label, round(probability * 100, 2)) for label, probability in zip(top_labels, top_probabilities)]

# Print the predicted tags and their probabilities
print("Predicted Tags:")
for label, probability in predicted_tags:
    print(label, "Probability:", probability, "%")

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

