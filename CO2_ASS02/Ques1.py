import pandas as pd
from sklearn.linear_model import LinearRegression
data = {
    'area': [45, 60, 75, 50, 90],
    'rooms': [1, 2, 2, 1, 3],
    'location': [0, 1, 1, 0, 1],   # 0 means village and 1 means town or city like place
    'rent': [12000, 25000, 32000, 14000, 45000]
}
df = pd.DataFrame(data)
X = df[['area', 'rooms', 'location']]
y = df['rent']
model = LinearRegression()
model.fit(X, y)
print(model.predict([[70, 2, 1]]))