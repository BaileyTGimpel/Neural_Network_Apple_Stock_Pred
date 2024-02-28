# Project Overview
Predicting Apple stock closing price trends from historical data with GRU and LSTM models from tensorflow and keras.  

## The Data  
Historical Apple Stock Data: https: //www.kaggle.com/datasets/camnugent/sandp500    

# Feature Engineering     
The data set AAPL_data.csv contains the columns: [date, open, high, low, close, volume, and name] the name column was dropped from the dataframe and the date column was converted to datetime and set as the index.  


## Creating Moving Averages 
In order to create features that will aid in the models predictions the 5 day and 50 day moving averages were calculated and stored in columns of the dataframe.  

```
df['mv_avg_short'] = df['close'].rolling(window=5).mean() 
df['mv_avg_long'] = df['close'].rolling(window=50).mean()  
```
## Feature Description: 
X (Independent Variables):

open: The price at which a stock first trades upon the opening of an exchange on a given trading day.
high: The highest price at which a stock trades during a specific period.
low: The lowest price of the stock during the same period.
volume: The number of shares or contracts traded in a security or an entire market during a given period.
mv_avg_short (Moving Average Short Term): This is a commonly used indicator in technical analysis that helps smooth out price action by filtering out the “noise” from random price fluctuations. It's a short-term moving average, possibly over days or weeks.
mv_avg_long (Moving Average Long Term): Similar to the short-term moving average but calculated over a longer period. This could be over several weeks or months, providing insights into the longer-term trend of the stock price. 

Y (Dependent Variable):
close: The final price at which a stock trades during a regular trading session. This is the target variable that the GRU model aims to predict. 

## Scaling and Cleaning  
In order to use volume as a feature in our models the values for all of our features must be scaled between 0 and 1 due to how large the values for volume are. Aditionally the first 50 days of data is removed due to the fact that we do not have 50 moving averages for those days. 

```
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = pd.DataFrame(scaler.fit_transform(df[["high","low","open","close","volume",
                                                 "mv_avg_short","mv_avg_long"]].values)) 
 
df_scaled = df_scaled.iloc[50:,:]
```
# Data Preprocessing  

## Formatting Data for GRU Model and LSTM Model 
The data is restructured into sequences to fit the GRU model's requirements, which is designed to process data in sequences or time steps. Here, each sequence (or sample) consists of 60 consecutive time steps, with each time step comprising the selected features (open, high, low, volume, mv_avg_short, and mv_avg_long). The sequences in X are used to predict the value of Y, the closing price, at the subsequent time step. By training the GRU model on this structured data, it learns to understand the temporal dynamics and the relationship between the various features over time to make accurate predictions about the closing price. 

```
# Need the data to be in the form [sample, time steps, features (dimension of each element)]
samples = 60 # Number of samples (in past)
steps = 1 # Number of steps (in future)
X = [] # X array
Y = [] # Y array
for i in range(df_scaled.shape[0] - samples): 
    X.append(df_scaled.iloc[i:i+samples,[0,1,2,4,5,6]].values) # Independent Samples
    Y.append(df_scaled.iloc[i+samples, [3]].values) # Dependent Samples
print('Training Data: Length is ',len(X[0:1][0]),': ', X[0:1])
print('Testing Data: Length is ', len(Y[0:1]),': ', Y[0:1])
```
Dimensions of X (1149, 60, 6) Dimensions of Y (1149, 1)  

The data is then split into training and testting sets with 90% of the data making up the training set (1034 days) and 10% of the data representing the test set (115 days): 
```
threshold = round(0.9 * X.shape[0])
trainX, trainY = X[:threshold], Y[:threshold]
testX, testY =  X[threshold:], Y[threshold:] 
print(threshold)
print('Training Length',trainX.shape, trainY.shape,'Testing Length:',testX.shape, testY.shape)
```
Training Length (1034, 60, 6) (1034, 1) Testing Length: (115, 60, 6) (115, 1)    

# GRU Model   


 



