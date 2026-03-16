import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset.csv")

X = data[['Hours']]
y = data['Scores']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(pd.DataFrame([[9]], columns=['Hours']))

print("Predicted score:", prediction[0])
