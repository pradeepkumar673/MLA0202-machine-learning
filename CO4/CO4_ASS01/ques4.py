import sklearn_crfsuite

sentences = [
    ["My", "name", "is", "John", "Doe", "and", "my", "order", "is", "12345", "for", "the", "iPhone", "14"],
    ["Hi", "I", "am", "Jane", "Smith", "order", "ID", "98765", "product", "MacBook", "Pro"]
]

labels = [
    ["O", "O", "O", "B-NAME", "I-NAME", "O", "O", "O", "O", "B-ORDER", "O", "O", "B-PRODUCT", "I-PRODUCT"],
    ["O", "O", "O", "B-NAME", "I-NAME", "O", "O", "B-ORDER", "O", "B-PRODUCT", "I-PRODUCT"]
]

def word2features(sent, i):
    word = sent[i]
    return {
        'word.lower()': word.lower(),
        'word.isdigit()': word.isdigit()
    }

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

X_train = [sent2features(s) for s in sentences]
y_train = labels

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
crf.fit(X_train, y_train)

test_sentence = ["Hello", "my", "name", "is", "Alice", "order", "54321", "item", "iPad"]
X_test = [sent2features(test_sentence)]

predicted_labels = crf.predict(X_test)[0]

print("Sentence:", test_sentence)
print("Predicted Labels:", predicted_labels)