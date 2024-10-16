----
layout: single
title: "2024 Samsung AI Challenge : Black-box Optimization 리더보드 4위"
----


# Samsung Blackbox Optimizer 경진대회


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
```

### 데이터 전처리 - LSTM


```python
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 훈련용 데이터 전처리
train_data.drop(columns=['ID'], inplace=True)
target = train_data.pop('y')

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(train_data)
y_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

timesteps = 10
X_lstm = []
y_lstm = []

for i in range(len(X_scaled) - timesteps):
    X_lstm.append(X_scaled[i:i + timesteps])
    y_lstm.append(y_scaled[i + timesteps])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# 테스트용 데이터 전처리
test_ids = test_data['ID']
test_data.drop(columns=['ID'], inplace=True)
test_data_scaled = scaler_X.transform(test_data)
```

### 모델 훈련 - ADAM, EarlyStopping


```python
def create_model(units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model(units=100, dropout_rate=0.3)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2)

os.makedirs('./model_saved/', exist_ok=True)
model.save('./model_saved/lstm_solved_model.h5')
```

### 모델 테스트


```python
y_test_pred_list = []

for i in range(len(test_data_scaled) - timesteps + 1):
    X_test_sample = np.array([test_data_scaled[i:i + timesteps]])
    y_test_pred_scaled = model.predict(X_test_sample)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_pred_list.append(y_test_pred.flatten()[0])

for i in range(1, timesteps):
    X_test_sample = np.array([test_data_scaled[-(timesteps + i):]])
    y_test_pred_scaled = model.predict(X_test_sample)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_pred_list.append(y_test_pred.flatten()[0])

output = pd.DataFrame({'ID': test_ids, 'y': y_test_pred_list})
output.to_csv('submission.csv', index=False)
```
