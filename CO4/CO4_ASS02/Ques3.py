import sklearn_crfsuite
from sklearn_crfsuite import metrics

train = [
    [("John","B-CUSTOMER"),("bought","O"),("Laptop","B-PRODUCT"),("1234","B-ORDER")],
    [("Priya","B-CUSTOMER"),("bought","O"),("Phone","B-PRODUCT"),("5678","B-ORDER")]
]

test = [
    [("Rahul","B-CUSTOMER"),("bought","O"),("Tablet","B-PRODUCT"),("9999","B-ORDER")]
]

def features(s):
    return [[
        "word=" + w,
        "lower=" + w.lower(),
        "digit=" + str(w.isdigit())
    ] for w, l in s]

X = [features(s) for s in train]
y = [[l for w, l in s] for s in train]

Xt = [features(s) for s in test]
yt = [[l for w, l in s] for s in test]

model = sklearn_crfsuite.CRF(max_iterations=100)
model.fit(X, y)

pred = model.predict(Xt)

for word, label in zip([w for w, l in test[0]], pred[0]):
    print(word, "->", label)

print("Accuracy:", metrics.flat_accuracy_score(yt, pred))