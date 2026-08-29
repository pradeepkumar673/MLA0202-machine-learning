import pandas as pd
from sklearn.linear_model import LogisticRegression
data = {
    'free': [5, 0, 4, 0, 3],
    'offer': [3, 0, 2, 1, 0],
    'money': [4, 0, 3, 0, 1],
    'spam': [1, 0, 1, 0, 0]   # 1 means spam and 0 means not spam
}
df = pd.DataFrame(data)
X = df[['free', 'offer', 'money']]
y = df['spam']
model = LogisticRegression()
model.fit(X, y)
print(model.predict([[4, 3, 2]]))