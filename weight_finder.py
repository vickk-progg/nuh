import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata = pd.read_csv("black balls.csv")
x = mydata[["height"]]
y = mydata[["weight"]]

pt.scatter(x,y)
pt.show()
model = lm.LinearRegression()
model.fit(x,y)
print(model.predict([[]]))