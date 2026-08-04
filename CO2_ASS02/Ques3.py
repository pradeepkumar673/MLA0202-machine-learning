import pandas as pd
from sklearn.naive_bayes import GaussianNB
data = {
    'fever': [1, 1, 0, 1, 0],
    'cough': [1, 0, 0, 1, 1],
    'headache': [1, 1, 0, 0, 0],
    'flu': [1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
X = df[['fever', 'cough', 'headache']]
y = df['flu']
model = GaussianNB()
model.fit(X, y)
print(model.predict([[1, 1, 0]]))