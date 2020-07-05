from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import yfinance as yf

data=yf.download('IDBIGOLD.NS','2019-07-05','2020-07-05')
print(data)
data=data[['Close']]
data=data.dropna()
data.Close.plot(figsize=(10,5))
plt.ylabel("IDBI GOLD ETF PRICES")
plt.show()
data['S_3'] = data['Close'].shift(1).rolling(window=3).mean()
data['S_9']= data['Close'].shift(1).rolling(window=9).mean()
data= data.dropna()
X = data[['S_3','S_9']]
X.head()
y = data['Close']
y.head()
t=.8
t = int(t*len(data))
# Train dataset
X_train = X[:t]
y_train = y[:t]
# Test dataset
X_test = X[t:]
y_test = y[t:]
linear = LinearRegression().fit(X_train,y_train)
print ("Gold ETF Price =", round(linear.coef_[0],2), \
"* 3 Days Moving Average", round(linear.coef_[1],2), \
"* 9 Days Moving Average +", round(linear.intercept_,2))
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['predicted_price','actual_price'])
plt.ylabel("Gold ETF Price(in indian Rs)")
plt.show()

r2_score = linear.score(X[t:],y[t:])*100
float("{0:.2f}".format(r2_score))