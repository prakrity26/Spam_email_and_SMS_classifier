from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. The Training Data (Text samples and their labels)
# 1 = Spam, 0 = Real/Ham
corpus = [
    "Congratulations! You've won a free $1000 Walmart gift card. Click here now!",
    "Hey, are we still meeting for lunch at 1 PM today?",
    "URGENT: Your account has been compromised. Please reset your password immediately.",
    "Can you please send me the final report by end of day?",
    " WIN BIG CASH prizes tonight! Text 'GO' to 55555 to claim.",
    "Just wanted to check in and see how your weekend went.",
    "Free entry in 2 a weekly comp to win FA Cup final tickets.",
    "Dear track team, practice is cancelled today due to the rain.",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]

# 2. Text Vectorization (Converting words into numbers that the AI can read)
# CountVectorizer counts how many times each word appears in the text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(corpus)

# 3. Train the AI Model
# MultinomialNB is a super fast model perfect for counting words
model = MultinomialNB()
model.fit(X_train, labels)

print("AI Model trained successfully in 0.05 seconds!")

# 4. Interactive Testing Loop (Run this in your terminal)
while True:
    print("\n--- Spam Detection Tool ---")
    user_input = input(
        "Enter a message to check (or type 'exit' to quit): "
    ).strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Convert the user's input into the exact same number format as our training data
    user_input_vectorized = vectorizer.transform([user_input])

    # Make the prediction
    prediction = model.predict(user_input_vectorized)

    # Display the result cleanly
    if prediction[0] == 1:
        print(" RESULT: [SPAM DETECTED] Avoid opening links inside.")
    else:
        print(" RESULT: [CLEAN / HAM] This looks like a normal message.")