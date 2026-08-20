import sklearn_crfsuite
from sklearn_crfsuite import metrics

train_data = [
    [
        ("John", "B-CUSTOMER"),
        ("ordered", "O"),
        ("a", "O"),
        ("Laptop", "B-PRODUCT"),
        ("with", "O"),
        ("order", "O"),
        ("12345", "B-ORDER"),
        ("and", "O"),
        ("reported", "O"),
        ("a", "O"),
        ("broken", "B-ISSUE"),
        ("screen", "I-ISSUE")
    ],
    [
        ("Priya", "B-CUSTOMER"),
        ("bought", "O"),
        ("Phone", "B-PRODUCT"),
        ("with", "O"),
        ("order", "O"),
        ("56789", "B-ORDER"),
        ("and", "O"),
        ("said", "O"),
        ("battery", "B-ISSUE"),
        ("is", "I-ISSUE"),
        ("dead", "I-ISSUE")
    ],
    [
        ("Rahul", "B-CUSTOMER"),
        ("received", "O"),
        ("Tablet", "B-PRODUCT"),
        ("under", "O"),
        ("order", "O"),
        ("24680", "B-ORDER"),
        ("and", "O"),
        ("reported", "O"),
        ("damaged", "B-ISSUE"),
        ("screen", "I-ISSUE")
    ],
    [
        ("Arun", "B-CUSTOMER"),
        ("purchased", "O"),
        ("Monitor", "B-PRODUCT"),
        ("for", "O"),
        ("order", "O"),
        ("67890", "B-ORDER"),
        ("and", "O"),
        ("reported", "O"),
        ("display", "B-ISSUE"),
        ("problem", "I-ISSUE")
    ]
]

test_data = [
    [
        ("Anita", "B-CUSTOMER"),
        ("ordered", "O"),
        ("Laptop", "B-PRODUCT"),
        ("under", "O"),
        ("order", "O"),
        ("13579", "B-ORDER"),
        ("and", "O"),
        ("reported", "O"),
        ("battery", "B-ISSUE"),
        ("problem", "I-ISSUE")
    ]
]

def word_features(sentence, index):
    word = sentence[index][0]

    features = {
        "word": word,
        "lowercase": word.lower(),
        "uppercase": word.isupper(),
        "digit": word.isdigit(),
        "length": len(word),
        "prefix": word[:2],
        "suffix": word[-2:]
    }

    if index > 0:
        previous_word = sentence[index - 1][0]
        features["previous_word"] = previous_word.lower()
    else:
        features["beginning"] = True

    if index < len(sentence) - 1:
        next_word = sentence[index + 1][0]
        features["next_word"] = next_word.lower()
    else:
        features["ending"] = True

    return features

def get_features(sentence):
    return [
        word_features(sentence, i)
        for i in range(len(sentence))
    ]

X_train = [
    get_features(sentence)
    for sentence in train_data
]

y_train = [
    [label for word, label in sentence]
    for sentence in train_data
]

X_test = [
    get_features(sentence)
    for sentence in test_data
]

y_test = [
    [label for word, label in sentence]
    for sentence in test_data
]

model = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    max_iterations=100,
    all_possible_transitions=True
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("CRF Named Entity Recognition")
print("----------------------------")

print("\nPredicted Entities:")

for i in range(len(test_data[0])):
    word = test_data[0][i][0]
    label = predictions[0][i]
    print(word, "->", label)

accuracy = metrics.flat_accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    metrics.flat_classification_report(
        y_test,
        predictions
    )
)