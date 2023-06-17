#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd ~/desktop


# In[2]:


pip install stargazer


# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import statsmodels.formula.api as smf
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
import yfinance as yf


# # Importing Bitcoin Data 

# In[4]:


BTC = yf.Ticker("BTC-USD")
start = '2018-01-01'
end = "2022-05-01"
BTC = BTC.history(start=start, end=end)
BTC["date"]=BTC.index
BTC = BTC[['Close','date']]
BTC


# # Importing Ethereum Data

# In[5]:


ETH=yf.Ticker("ETH-USD")
start = '2018-01-01'
end = "2022-05-01"
ETH = ETH.history(start=start, end=end)
ETH["date"]=ETH.index
ETH = ETH[['Close','date']]
ETH


# # Importing Solana Data

# In[6]:


SOL =yf.Ticker("SOL-USD")
start = '2018-01-01'
end = "2022-05-01"
SOL = SOL.history(start=start, end=end)
SOL["date"]=SOL.index
SOL = SOL[['Close','date']]
SOL


# # Importing Cardano Data

# In[7]:


ADA =yf.Ticker("ADA-USD")
start = '2018-01-01'
end = "2022-05-01"
ADA = ADA.history(start=start, end=end)
ADA["date"]= ADA.index
ADA = ADA[['Close','date']]
ADA


# # Importing Binance Coin Data

# In[8]:


BIN =yf.Ticker("BNB-USD")
start = '2018-01-01'
end = "2022-05-01"
BIN = BIN.history(start=start, end=end)
BIN["date"]= BIN.index
BIN = BIN[['Close','date']]
BIN


# In[9]:


XRP =yf.Ticker("XRP-USD")
start = '2018-01-01'
end = "2022-05-01"
XRP = XRP.history(start=start, end=end)
XRP["date"]= XRP.index
XRP = XRP[['Close','date']]
XRP


# # Calculating Percentage Change Per Closing Price

# In[10]:


BTC['BTCPCH'] = BTC['Close'].pct_change()*100
BTC


# In[11]:


ETH['ETHPCH'] = ETH['Close'].pct_change()*100
ETH


# In[12]:


SOL['SOLPCH'] = SOL['Close'].pct_change()*100
SOL


# In[13]:


ADA['ADAPCH'] = ADA['Close'].pct_change()*100
ADA


# In[14]:


BIN['BINPCH'] = BIN['Close'].pct_change()*100
BIN


# In[15]:


XRP['XRPPCH'] = XRP['Close'].pct_change()*100
XRP


# # Creating a new data frame with all Percentage Changes for each CryptoAsset

# In[17]:


data= [BTC["BTCPCH"],ETH['ETHPCH'],SOL['SOLPCH'],ADA['ADAPCH'],BIN['BINPCH'],XRP['XRPPCH']]
headers = ['BTC','ETH','SOL','ADA','BIN','XRP']
data = pd.concat(data,axis=1,keys=headers)
data = data.dropna()
data


# In[18]:


data.corr()


# # Regression with ETH & Bitcoin

# In[19]:


import statsmodels.api as sm
X = data['ETH']
y = data['BTC']
X = sm.add_constant(X)
ETHmodel = sm.OLS(y,X)
results = ETHmodel.fit()
results.summary()


# In[20]:


fig = plt.figure(figsize=(20, 8))
plt.plot(data.index, data['ETH'], color='blue',label='ETH', 
         linewidth=2)
plt.plot(data.index, data['BTC'], color='orange', 
         label='BTC', linewidth=2)

# Add title and labels

plt.title('Bitcoin & ETH Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percetage Change')

# Add legend

plt.legend()

# Auto space

plt.tight_layout()

# Display plot

plt.show() 


# # Regression with SOL & Bitcoin

# In[21]:


X = data['SOL']
y = data['BTC']
X = sm.add_constant(X)
SOLmodel = sm.OLS(y,X)
results = SOLmodel.fit()
results.summary()


# In[22]:


fig = plt.figure(figsize=(20, 8))
plt.plot(data.index, data['SOL'], color='purple',label='SOL', 
         linewidth=2)
plt.plot(data.index, data['BTC'], color='orange', 
         label='BTC', linewidth=2)

# Add title and labels

plt.title('Bitcoin & SOL Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percetage Change')

# Add legend

plt.legend()

# Auto space

plt.tight_layout()

# Display plot

plt.show() 


# # Regression with ADA & Bitcoin

# In[23]:


X = data['ADA']
y = data['BTC']
X = sm.add_constant(X)
ADAmodel = sm.OLS(y,X)
results = ADAmodel.fit()
results.summary()


# In[24]:


fig = plt.figure(figsize=(20, 8))
plt.plot(data.index, data['ADA'], color='teal',label='ADA', 
         linewidth=2)
plt.plot(data.index, data['BTC'], color='orange', 
         label='BTC', linewidth=2)

# Add title and labels

plt.title('Bitcoin & ADA Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percetage Change')

# Add legend

plt.legend()

# Auto space

plt.tight_layout()

# Display plot

plt.show() 


# # Regression on Binance Coin & Bitcoin

# In[25]:


X = data['BIN']
y = data['BTC']
X = sm.add_constant(X)
BINmodel = sm.OLS(y,X)
results = ADAmodel.fit()
results.summary()


# In[26]:


fig = plt.figure(figsize=(20, 8))
plt.plot(data.index, data['BIN'], color='yellow',label='BIN', 
         linewidth=2)
plt.plot(data.index, data['BTC'], color='red', 
         label='BTC', linewidth=2)

# Add title and labels

plt.title('Bitcoin & BIN Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percetage Change')

# Add legend

plt.legend()

# Auto space

plt.tight_layout()

# Display plot

plt.show() 


# # Regression with XRP and Bitcoin

# In[27]:


X = data['XRP']
y = data['BTC']
X = sm.add_constant(X)
XRPmodel = sm.OLS(y,X)
results = XRPmodel.fit()
results.summary()


# In[28]:


fig = plt.figure(figsize=(20, 8))
plt.plot(data.index, data['XRP%'], color='black',label='XRP', 
         linewidth=2)
plt.plot(data.index, data['BTC%'], color='red', 
         label='BTC', linewidth=2)

# Add title and labels

plt.title('Bitcoin & XRP Percentage Change')
plt.xlabel('Date')
plt.ylabel('Percetage Change')

# Add legend

plt.legend()

# Auto space

plt.tight_layout()

# Display plot

plt.show() 


# #  Regression to predict Bitcoin's movements from the price change of other assets (The Linear Model)

# In[29]:


BTC_Linear_Model = smf.ols(formula="BTC~ETH+SOL+ADA+BIN+XRP", data=data).fit()
print(BTC_Linear_Model.summary())


# In[30]:


stargazer = Stargazer([model1])
print(stargazer.render_latex())


# In[31]:


for table in model1.summary().tables:
    print(table.as_latex_tabular())


# In[32]:


stargazer


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1000)
y_predict = BTC_Linear_Model.predict(X_test)
print(y_predict)


# In[ ]:


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_predict)
MSE


# In[ ]:


RMSE = np.sqrt(MSE)
RMSE


# # Data Preprocessing for the Polynomial Regression

# In[ ]:





# In[ ]:


data= [BTC["BTCPCH"],ETH['ETHPCH'],SOL['SOLPCH'],ADA['ADAPCH'],BIN['BINPCH'],XRP['XRPPCH']]
headers = ['BTC','ETH','SOL','ADA','BIN','XRP']
data = pd.concat(data,axis=1,keys=headers)
data


# In[ ]:


data.describe()


# In[ ]:


poly = smf.ols(formula="BTC~I(np.power(ETH,5))+I(np.power(SOL,5))+I(np.power(ADA,5))+I(np.power(BIN,5))+I(np.power(XRP,5))", data=data).fit()
print(poly.summary())


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1000)
y_predict = poly.predict(X_test)
print(y_predict)


# In[ ]:


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_predict)
MSE


# In[ ]:


RMSE = np.sqrt(MSE)
RMSE


# # The Random Forest Regression Model

# In[ ]:


X = data[['ETH','ADA','XRP','SOL','BIN']]
y = data['BTC']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1000)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
rf = RandomForestRegressor()
rf_model = rf.fit(X_train,y_train)


# In[ ]:


y_predict = rf.predict(X_test)
print(y_predict)


# In[ ]:


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_predict)
MSE


# In[ ]:


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_predict)
MSE


# In[ ]:


RMSE = np.sqrt(MSE)
RMSE


# In[ ]:


R_Squared = 1 - sum((y_predict)**2)/sum((y-np.mean(y))**2)
R_Squared


# In[ ]:


rf_model.feature_importances_


# In[ ]:





# In[ ]:





# In[ ]:




