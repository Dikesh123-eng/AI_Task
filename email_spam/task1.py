import pandas as pd

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv")

# Convert labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and target
X = data['text']
y = data['label']

# Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🔮 Prediction
while True:
    msg = input("\nEnter message: ")
    
    if msg.lower() == "exit":
        break

    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)[0]

    if result == 1:
        print("🚫 Spam Message")
    else:
        print("✅ Not Spam")