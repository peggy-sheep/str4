# import packages

import streamlit as st
import pandas as pd
import numpy as np # this package is for data calculation
import matplotlib.pyplot as plt # for data visualization 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import torch
from torch import nn
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

#=======================================================================================================================

# Side Panel
df = pd.read_csv('./function11_1.csv')
df2 = pd.read_csv('./function11_1.csv',parse_dates=['Start_Date'],index_col='Start_Date')
df3 = pd.read_csv('./function11_2.csv',parse_dates = ['Start_Date'] )

df4 = pd.read_csv('./function11_1.csv')
df4 = df4.drop(labels=['carpark','home','other','street','work','start_temperature','id'],axis='columns')
df4.set_index('Start_Date' , inplace=True)
print(df4)

df6=pd.read_csv('./function11_2.csv')
df6= df6.set_index("Start_Date")
df6.index=pd.to_datetime(df6.index)

df7 = pd.read_csv('./function11_2.csv', index_col=[0], parse_dates=[0])
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]

reload_model_bilstm = tf.keras.models.load_model('./keras_model_bilstm_3.h5')
reload_model_bilstm.summary()

reload_model_gru= tf.keras.models.load_model('./keras_model_gru_3.h5')
reload_model_gru.summary()

reload_model_MLP = tf.keras.models.load_model('./model_MLP.h5')
reload_model_MLP.summary()
#====================================================================================================================

st.sidebar.header("Predict Part")
selected_status_6 = st.sidebar.selectbox('Select model', options = ["GRU", "Bi-LSTM", "LSTM", "CNN-LSTM","XGBoost","MLP","Transformer"])
selected_status_7 = st.sidebar.slider('Predict days', min_value=7, max_value=28, step=7)

st.sidebar.write('Evaluation model')
selected_status_8_1 = st.sidebar.checkbox('RMSE')
selected_status_8_2 = st.sidebar.checkbox('MSE')
selected_status_8_3 = st.sidebar.checkbox('MAE')


# ===================================================================================================================
if selected_status_6 == 'GRU':

    tf.random.set_seed(1234)

    # Check for missing values
    print('Total num of missing values:') 
    print(df3.Total_kWh.isna().sum())
    print(' ')
    # Locate the missing value
    df_missing_date = df3.loc[df3.Total_kWh.isna() == True]
    print('The date of missing value:')
    print(df_missing_date.loc[:,['Start_Date']])
    # Replcase missing value with interpolation
    df3.Total_kWh.interpolate(inplace = True)
    # Keep WC and drop Date
    df3 = df3.drop('Start_Date', axis = 1)

    # Split train data and test data
    train_size = int(len(df3)*0.8)

    train_data = df3.iloc[:train_size]
    test_data = df3.iloc[train_size:]

    scaler = MinMaxScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    print(train_data.shape)
    print(test_data.shape)

    # Create input dataset
    def create_dataset (X, look_back = 1):
        Xs, ys = [], []
    
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
    
        return np.array(Xs), np.array(ys)
    LOOK_BACK = 7
    X_train, y_train = create_dataset(train_scaled,LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled,LOOK_BACK)
    # Print data shape
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape) 
    print('y_test.shape: ', y_test.shape)

    print(X_test[:33].shape)

    y_test = scaler.inverse_transform(y_test)
    y_train = scaler.inverse_transform(y_train)

    if selected_status_7 == 7:
        new_data = test_data.iloc[:14]
        train_data = train_data[282:]
    elif selected_status_7 == 14:
        new_data = test_data.iloc[:21]
        train_data = train_data[275:]
    elif selected_status_7 == 21:
        new_data = test_data.iloc[:28]
        train_data = train_data[268:]
    elif selected_status_7 == 28:
        new_data = test_data.iloc[:35]
        train_data = train_data[261:]

    # Select 60 days of data from test data
    
    # Scale the input
    scaled_data = scaler.transform(new_data)
    # Reshape the input 
    def create_dataset (X, look_back = 7):
        Xs = []
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            
        return np.array(Xs)

    X_30= create_dataset(scaled_data,7)
    print('X_30.shape: ', X_30.shape)

    st.subheader("Predict Part")

    # Make prediction for new data
    def prediction(model):
        prediction = model.predict(X_30)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_gru = prediction(reload_model_gru)
   
    # Plot history and future
    def plot_multi_step(history, prediction1):
        
        fig4 = plt.figure(figsize=(15, 6))
        
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))

        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with GRU')
        plt.legend(loc='upper right')
        plt.xlabel('Time step')
        plt.ylabel('Tital_kwh')
        st.pyplot(fig4)
    plot_multi_step(train_data, prediction_gru)

    if selected_status_7 == 7:
        st.write('Predicted value of charging energy consumption in the next 7 days')
    elif selected_status_7 == 14:
        st.write('Predicted value of charging energy consumption in the next 14 days')
    elif selected_status_7 == 21:
        st.write('Predicted value of charging energy consumption in the next 21 days')
    else:
        st.write('Predicted value of charging energy consumption in the next 28 days')
    
    st.write(prediction_gru)

    # Make prediction
    def prediction(model):
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_gru = prediction(reload_model_gru)


    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    evaluate_prediction(prediction_gru, y_test,'GRU')
    

#=======================================================================================================================
elif selected_status_6 == 'Bi-LSTM':
    tf.random.set_seed(1234)

    # Check for missing values
    print('Total num of missing values:') 
    print(df3.Total_kWh.isna().sum())
    print(' ')
    # Locate the missing value
    df_missing_date = df3.loc[df3.Total_kWh.isna() == True]
    print('The date of missing value:')
    print(df_missing_date.loc[:,['Start_Date']])
    # Replcase missing value with interpolation
    df3.Total_kWh.interpolate(inplace = True)
    # Keep WC and drop Date
    df3 = df3.drop('Start_Date', axis = 1)

    # Split train data and test data
    train_size = int(len(df3)*0.8)

    train_data = df3.iloc[:train_size]
    test_data = df3.iloc[train_size:]

    scaler = MinMaxScaler().fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    print(train_data.shape)

    # Create input dataset
    def create_dataset (X, look_back = 1):
        Xs, ys = [], []
    
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            ys.append(X[i+look_back])
        return np.array(Xs), np.array(ys)

    LOOK_BACK = 7
    X_train, y_train = create_dataset(train_scaled,LOOK_BACK)
    X_test, y_test = create_dataset(test_scaled,LOOK_BACK)

    # Print data shape
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape) 
    print('y_test.shape: ', y_test.shape)

    print(X_test[:33].shape)

    y_test = scaler.inverse_transform(y_test)
    y_train = scaler.inverse_transform(y_train)

    if selected_status_7 == 7:
        new_data = test_data.iloc[:14]
        train_data = train_data[282:]
    elif selected_status_7 == 14:
        new_data = test_data.iloc[:21]
        train_data = train_data[275:]
    elif selected_status_7 == 21:
        new_data = test_data.iloc[:28]
        train_data = train_data[268:]
    elif selected_status_7 == 28:
        new_data = test_data.iloc[:35]
        train_data = train_data[261:]


    # Select 60 days of data from test data
    # Scale the input
    scaled_data = scaler.transform(new_data)
    # Reshape the input 
    def create_dataset (X, look_back = 7):
        Xs = []
        for i in range(len(X)-look_back):
            v = X[i:i+look_back]
            Xs.append(v)
            
        return np.array(Xs)

    X_30= create_dataset(scaled_data,7)
    print('X_30.shape: ', X_30.shape)
    
    st.subheader("Predict Part")

    # Make prediction for new data
    def prediction(model):
        prediction = model.predict(X_30)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_bilstm = prediction(reload_model_bilstm)
    

    # Plot history and future
    def plot_multi_step(history, prediction1):
        fig4 = plt.figure(figsize=(15, 6))
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))
        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with BiLSTM')
        plt.legend(loc='upper right')
        plt.xlabel('Time step')
        plt.ylabel('Tital_kwh') 
        st.pyplot(fig4)
    plot_multi_step(train_data, prediction_bilstm)

    if selected_status_7 == 7:
        st.write('Predicted value of charging energy consumption in the next 7 days')
    elif selected_status_7 == 14:
        st.write('Predicted value of charging energy consumption in the next 14 days')
    elif selected_status_7 == 21:
        st.write('Predicted value of charging energy consumption in the next 21 days')
    else:
        st.write('Predicted value of charging energy consumption in the next 28 days')
    
    st.write(prediction_bilstm)

    # Make prediction
    def prediction(model):
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_bilstm = prediction(reload_model_bilstm)
        

    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    
    evaluate_prediction(prediction_bilstm, y_test, 'Bidirectiona LSTM')

    # model_gru.save('./keras_model_gru_3.h5')
    # model_bilstm.save('./keras_model_bilstm_3.h5')
# ===========================================================================================================================
elif selected_status_6 == 'CNN-LSTM':
    st.subheader("Predict Part")
    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(df4)-seq_length):
            x = data.iloc[i:(i+seq_length)]
            y = data.iloc[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Data transformation for supervised learning data.
    seq_length = 7
    X, y = create_sequences(df4, seq_length)

    # Dividing the dataset into traning, validation, and test sets.
    train_size = int(363 * 0.8)
    X_train = X[:train_size]
    X_val, y_val = X[train_size:train_size+33], y[train_size:train_size+33]
    print("X_train",X_train)
    
    

    if selected_status_7 == 7:
        y_train = y[train_size-15:train_size-7]
        X_test, y_test = X[train_size:train_size+7], y[train_size:train_size+7]
    elif selected_status_7 == 14:
        y_train = y[train_size-22:train_size-7]
        X_test, y_test = X[train_size:train_size+14], y[train_size:train_size+14]
    elif selected_status_7 == 21:
        y_train = y[train_size-29:train_size-7]
        X_test, y_test = X[train_size:train_size+21], y[train_size:train_size+21]
    elif selected_status_7 == 28:
        y_train = y[train_size-37:train_size-7]
        X_test, y_test = X[train_size:train_size+28], y[train_size:train_size+28]

    # print(X_test)
    # print(train_size)

    MIN = X_train.min()
    MAX = X_train.max()

    def MinMaxScale(array, min, max):

        return (array - min) / (max - min)

    # MinMax scaling
    X_train = MinMaxScale(X_train, MIN, MAX)
    y_train = MinMaxScale(y_train, MIN, MAX)
    X_val = MinMaxScale(X_val, MIN, MAX)
    y_val = MinMaxScale(y_val, MIN, MAX)
    X_test = MinMaxScale(X_test, MIN, MAX)
    y_test = MinMaxScale(y_test, MIN, MAX)

    # Tensor transformation
    def make_Tensor(array):
        return torch.from_numpy(array).float()

    X_train = make_Tensor(X_train)
    y_train = make_Tensor(y_train)
    X_val = make_Tensor(X_val)
    y_val = make_Tensor(y_val)
    X_test = make_Tensor(X_test)
    y_test = make_Tensor(y_test)

    c = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
    input = torch.Tensor([[[1,2,3,4,5,6,7]]])
    output = c(input)
    # print(output)

    for param in c.parameters():
        print(param)

    w_list = []
    for param in c.parameters():
        w_list.append(param)

    w = w_list[0]
    b = w_list[1]

    w1 = w[0][0][0]
    w2 = w[0][0][1]

    print(w1)
    print(w2)
    print(b)

    w1 * 3 + w2 * 4 + b

    # print(output)

    class CovidPredictor(nn.Module):
        def __init__(self, n_features, n_hidden, seq_len, n_layers):
            super(CovidPredictor, self).__init__()
            self.n_hidden = n_hidden
            self.seq_len = seq_len
            self.n_layers = n_layers
            self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1) # Add a 1D CNN layer
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=n_hidden,
                num_layers=n_layers
            )
            self.linear = nn.Linear(in_features=n_hidden, out_features=1)
        def reset_hidden_state(self):
            self.hidden = (
                torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden),
                torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden)
            )
        def forward(self, sequences):
            sequences = self.c1(sequences.view(len(sequences), 1, -1))
            lstm_out, self.hidden = self.lstm(
                sequences.view(len(sequences), self.seq_len-1, -1),
                self.hidden
            )
            last_time_step = lstm_out.view(self.seq_len-1, len(sequences), self.n_hidden)[-1]
            y_pred = self.linear(last_time_step)
            return y_pred

    def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100, verbose = 10, patience = 10):
        loss_fn = torch.nn.L1Loss() #
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        train_hist = []
        val_hist = []
        for t in range(num_epochs):

            epoch_loss = 0

            for idx, seq in enumerate(train_data): # hidden state needs to be reset after every sample

                model.reset_hidden_state()

                # train loss
                seq = torch.unsqueeze(seq, 0)
                y_pred = model(seq)
                loss = loss_fn(y_pred[0].float(), train_labels[idx]) # calculated loss after 1 step

                # update weights
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            train_hist.append(epoch_loss / len(train_data))

            if val_data is not None:

                with torch.no_grad():

                    val_loss = 0

                    for val_idx, val_seq in enumerate(val_data):

                        model.reset_hidden_state() # hidden state reset every sequence

                        val_seq = torch.unsqueeze(val_seq, 0)
                        y_val_pred = model(val_seq)
                        val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])

                        val_loss += val_step_loss
                    
                val_hist.append(val_loss / len(val_data)) # append in val hist

                ## print loss for every `verbose` times
                if t % verbose == 0:
                    print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

                ## check early stopping for every `patience` times
                if (t % patience == 0) & (t != 0):
                    
                    ## if loss increased, perform early stopping
                    if val_hist[t - patience] < val_hist[t] :

                        print('\n Early Stopping')

                        break

            # elif t % verbose == 0:
            #     print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')

                
        return model, train_hist, val_hist

    model = CovidPredictor(
        n_features=1,
        n_hidden=4,
        seq_len=seq_length,
        n_layers=1
    )
    print(model)

    FILE = "model.pth"
    model = torch.load(FILE)
    model.eval()

    pred_dataset = X_test

    with torch.no_grad():
        preds = []
        for _ in range(len(pred_dataset)):
            model.reset_hidden_state()
            y_test_pred = model(torch.unsqueeze(pred_dataset[_], 0))
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)

    print(np.array(preds)*MAX)


    def plot_multi_step(history, prediction1):
        fig5 = plt.figure(figsize=(15, 6))
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))
        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with CNN-LSTM')
        plt.legend(loc='upper right')
        plt.xlabel('days')
        plt.ylabel('Tital_kwh') 
        st.pyplot(fig5)
    plot_multi_step(np.array(y_train)*MAX, np.array(preds)*MAX)

    if selected_status_7 == 7:
        st.write('Predicted value of charging energy consumption in the next 7 days')
    elif selected_status_7 == 14:
        st.write('Predicted value of charging energy consumption in the next 14 days')
    elif selected_status_7 == 21:
        st.write('Predicted value of charging energy consumption in the next 21 days')
    else:
        st.write('Predicted value of charging energy consumption in the next 28 days')
    
    st.write(np.array(preds)*MAX)

    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    evaluate_prediction(np.array(preds)*MAX, np.array(y_test)*MAX ,'CNN-LSTM')

#=================================================================================================================================
elif selected_status_6 == 'LSTM':
    st.subheader("Predict Part")
    test_data_size = 73

    all_data = df3['Total_kWh'].values.astype(float)

    train_data_X = all_data[:-test_data_size]
    train_data_y = all_data[:-test_data_size]
    test_data_X = all_data[-test_data_size:]
    test_data_y = test_data_X

    print(len(train_data_X))
    print(len(test_data_X))

    print(test_data_X)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data_X .reshape(-1, 1))

    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
    train_window = 7
    # if selected_status_7 == 7:
    #     train_window = 7
    # elif selected_status_7 == 14:
    #     train_window = 14
    # elif selected_status_7 == 21:
    #     train_window = 21
    # elif selected_status_7 == 28:
    #     train_window = 28

    if selected_status_7 == 7:
        train_data_y = all_data[test_data_size + 209:-test_data_size]
    elif selected_status_7 == 14:
        train_data_y = all_data[test_data_size + 202:-test_data_size]
    elif selected_status_7 == 21:
        train_data_y = all_data[test_data_size + 195:-test_data_size]
    elif selected_status_7 == 28:
        train_data_y = all_data[test_data_size + 188:-test_data_size]

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                                torch.zeros(1,1,self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)

    # epochs = 200

    # for i in range(epochs):
    #     for seq, labels in train_inout_seq:
    #         optimizer.zero_grad()
    #         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
    #                         torch.zeros(1, 1, model.hidden_layer_size))

    #         y_pred = model(seq)

    #         single_loss = loss_function(y_pred, labels)
    #         single_loss.backward()
    #         optimizer.step()

    #     if i%25 == 1:
    #         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    FILE = "model_LSTM.pth"
    model = torch.load(FILE)
    model.eval()

    if selected_status_7 == 7:
        fut_pred = 7
    elif selected_status_7 == 14:
        fut_pred = 14
    elif selected_status_7 == 21:
        fut_pred = 21
    elif selected_status_7 == 28:
        fut_pred = 28
    
    test_inputs = train_data_normalized[-train_window:].tolist()
    print(test_inputs)

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())
        
    print(test_inputs[fut_pred:])

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))


    x = np.arange(333, 363, 1)
    print(x)

    # plt.title('Days vs Total_kwh')
    # plt.ylabel('Total_kwh')
    # plt.grid(True)
    # plt.autoscale(axis='x', tight=True)
    # plt.plot(df['Total_kWh'])
    # plt.plot(x,actual_predictions)
    # plt.show()

    # plt.title('Days vs Total_kwh')
    # plt.ylabel('Total_kwh')
    # plt.grid(True)
    # plt.autoscale(axis='x', tight=True)

    # plt.plot(df['Total_kWh'][-train_window:])
    # plt.plot(x,actual_predictions)
    # plt.show()

    def plot_multi_step(history, prediction1):
        fig6 = plt.figure(figsize=(15, 6))
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))
        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with LSTM')
        plt.legend(loc='upper right')
        plt.xlabel('days')
        plt.ylabel('Tital_kwh') 
        st.pyplot(fig6)
    plot_multi_step(train_data_y, actual_predictions)

    if selected_status_7 == 7:
        st.write('Predicted value of charging energy consumption in the next 7 days')
    elif selected_status_7 == 14:
        st.write('Predicted value of charging energy consumption in the next 14 days')
    elif selected_status_7 == 21:
        st.write('Predicted value of charging energy consumption in the next 21 days')
    else:
        st.write('Predicted value of charging energy consumption in the next 28 days')
    st.write(actual_predictions)

    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    evaluate_prediction(actual_predictions, test_data_y,'LSTM')

    
#=================================================================================================================================
elif selected_status_6 == 'MLP':
    st.subheader("Predict Part")

    n_train=int(len(df6)*0.8)
    n_test=len(df6)-n_train
    train,test=df6.iloc[0:n_train],df6.iloc[n_train:len(df6)]
    print(len(train),len(test))

    def get_timeseries(x,n_steps=1):
        x_ts,y_ts=[],[]
        for ii in range(len(x)-n_steps):
            v=x.iloc[ii:(ii+n_steps)].values
            x_ts.append(v)
            w=x.iloc[ii+n_steps].values
            y_ts.append(w)
        return np.array(x_ts),np.array(y_ts) 

    n_steps=1
    n_features=1

    x_train,y_train=get_timeseries(train,n_steps)
    x_test,y_test=get_timeseries(test,n_steps)

    print(x_train.shape,y_train.shape)

    if selected_status_7 == 7:
        y_train = y_train[281:]
        x_test = x_test[:7]
        y_test = y_test[:7]
    elif selected_status_7 == 14:
        y_train = y_train[274:]
        x_test = x_test[:14]
        y_test = y_test[:14]
    elif selected_status_7 == 21:
        y_train = y_train[267:]
        x_test = x_test[:21]
        y_test = y_test[:21]
    elif selected_status_7 == 28:
        y_train = y_train[260:]
        x_test = x_test[:28]
        y_test = y_test[:28]

    # model = Sequential()
    # model.add(Dense(100, activation='relu', input_dim=n_steps))
    # model.add(Dense(1))
    # model.compile(optimizer=Adam(0.001),loss='mse')
    # history=model.fit(x_train, y_train, epochs=100,batch_size=32,validation_split=0.1, verbose=0,shuffle=False)
    
    def prediction(model):
        prediction = model.predict(x_test)
        # prediction = scaler.inverse_transform(prediction)
        return prediction
    prediction_MLP = prediction(reload_model_MLP)
    


    # fig7 = plt.figure(figsize=(15, 6))
    # plt.title('50 epochs prediction analyze',fontsize=20)
    # plt.plot(y_test,marker='.',label='actual data')
    # plt.plot(prediction_MLP,'r',label='prediction')
    # plt.ylabel('kwh',fontsize=17)
    # plt.xlabel('period',fontsize=17)
    # plt.legend()
    # plt.show()

    def plot_multi_step(history, prediction1):
        fig7 = plt.figure(figsize=(15, 6))
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))
        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with MLP')
        plt.legend(loc='upper right')
        plt.xlabel('days')
        plt.ylabel('Tital_kwh') 
        st.pyplot(fig7)
    plot_multi_step(y_train, prediction_MLP)
    
    if selected_status_7 == 7:
        st.write('Predicted value of charging energy consumption in the next 7 days')
    elif selected_status_7 == 14:
        st.write('Predicted value of charging energy consumption in the next 14 days')
    elif selected_status_7 == 21:
        st.write('Predicted value of charging energy consumption in the next 21 days')
    else:
        st.write('Predicted value of charging energy consumption in the next 28 days')
    st.write(prediction_MLP)

    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    evaluate_prediction(prediction_MLP, y_test,'MLP')


# st.sidebar.header("Predict Part")
# st.subheader("Predict Part")

#=================================================================================================================================
elif selected_status_6 == 'XGBoost':
    st.subheader("Predict Part")
    split_date = '06-19-2018'
    df_train = df7.loc[df7.index <= split_date].copy()
    df_test = df7.loc[df7.index > split_date].copy()


    def create_features(df7, label=None):
        """
        Creates time series features from datetime index
        """
        df7['date'] = df7.index
        df7['dayofweek'] = df7['date'].dt.dayofweek
        df7['quarter'] = df7['date'].dt.quarter
        df7['month'] = df7['date'].dt.month
        
        X = df7[['dayofweek','quarter','month']]
        if label:
            y = df7[label]
            return X, y
        return X

    X_train, y_train = create_features(df_train, label='Total_kWh')
    X_test, y_test = create_features(df_test, label='Total_kWh')
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    
    
    # mask1 = y_train ["Team"] != "Marketing"


    print(y_train)
    print(X_test)

    # if selected_status_7 == 7:
    #     y_train = y[train_size-22:train_size-7]
    #     X_test, y_test = X[train_size:train_size+7], y[train_size:train_size+7]
    # elif selected_status_7 == 14:
    #     y_train = y[train_size-22:train_size-7]
    #     X_test, y_test = X[train_size:train_size+14], y[train_size:train_size+14]
    # elif selected_status_7 == 21:
    #     y_train = y[train_size-29:train_size-7]
    #     X_test, y_test = X[train_size:train_size+21], y[train_size:train_size+21]
    # elif selected_status_7 == 28:
    #     y_train = y[train_size-37:train_size-7]
    #     X_test, y_test = X[train_size:train_size+28], y[train_size:train_size+28]

    reg = xgb.XGBRegressor(n_estimators=150,learning_rate = 0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=15,
            eval_metric=["error", "logloss"],
            verbose=False) # Change verbose to True if you want to see it train

    #_ = plot_importance(reg, height=0.9)
    #plt.show()

    df_test['Prediction'] = reg.predict(X_test)
    print(df_test['Prediction'] )
    df_all = pd.concat([df_test, df_train], sort=False)
    #_ = df_all[['Total_kWh','MW_Prediction']].plot(figsize=(15, 5))
    #plt.show()
    if selected_status_7 == 7:
        split_date_7 = '06-12-2018'
        split_date_71 = '06-19-2018'
        split_date_72 = '06-20-2018'
        split_date_73 = '06-26-2018'
        
        y_train = y_train.loc[(y_train.index >= split_date_7)&(y_train.index <= split_date_71)]
        df_test['Prediction'] = df_test['Prediction'].loc[(df_test['Prediction'].index >= split_date_72)&(df_test['Prediction'].index <= split_date_73)]
    elif selected_status_7 == 14:
        split_date_14 = '06-05-2018'
        split_date_141 = '06-19-2018'
        split_date_142 = '06-20-2018'
        split_date_143 = '07-03-2018'
        
        y_train = y_train.loc[(y_train.index >= split_date_14)&(y_train.index <= split_date_141)]
        df_test['Prediction'] = df_test['Prediction'].loc[(df_test['Prediction'].index >= split_date_142)&(df_test['Prediction'].index <= split_date_143)]
    elif selected_status_7 == 21:
        split_date_21 = '05-29-2018'
        split_date_211 = '06-19-2018'
        split_date_212 = '06-20-2018'
        split_date_213 = '07-10-2018'
        
        y_train = y_train.loc[(y_train.index >= split_date_21)&(y_train.index <= split_date_211)]
        df_test['Prediction'] = df_test['Prediction'].loc[(df_test['Prediction'].index >= split_date_212)&(df_test['Prediction'].index <= split_date_213)]
    elif selected_status_7 == 28:
        split_date_28 = '05-22-2018'
        split_date_281 = '06-19-2018'
        split_date_282 = '06-20-2018'
        split_date_283 = '07-17-2018'
        
        y_train = y_train.loc[(y_train.index >= split_date_28)&(y_train.index <= split_date_281)]
        df_test['Prediction'] = df_test['Prediction'].loc[(df_test['Prediction'].index >= split_date_282)&(df_test['Prediction'].index <= split_date_283)]  
    
    def plot_multi_step(history, prediction1):
        fig4 = plt.figure(figsize=(15, 6))
        range_history = len(history)
        range_future = list(range(range_history, range_history + len(prediction1)))
        plt.plot(np.arange(range_history), np.array(history), label='History')
        plt.plot(range_future, np.array(prediction1),label='Forecasted with BiLSTM')
        plt.legend(loc='upper right')
        plt.xlabel('Time step')
        plt.ylabel('Tital_kwh') 
        st.pyplot(fig4)
        # plt.show()
    plot_multi_step(y_train, df_test['Prediction'] )

    if selected_status_7 == 7:
        st.write('Predicted value of charging energy consumption in the next 7 days')
        st.write(df_test['Prediction'].head(7))
    elif selected_status_7 == 14:
        st.write('Predicted value of charging energy consumption in the next 14 days')
        st.write(df_test['Prediction'].head(14))
    elif selected_status_7 == 21:
        st.write('Predicted value of charging energy consumption in the next 21 days')
        st.write(df_test['Prediction'].head(21))
    else:
        st.write('Predicted value of charging energy consumption in the next 28 days')
        st.write(df_test['Prediction'].head(28))

    def evaluate_prediction(predictions, actual, model_name):
        errors = predictions - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        st.write(model_name +' Evaluation Metrics(Please choose selected box)'+ ':')
        if selected_status_8_1:
            st.write('RMSE: {:.4f}'.format(rmse))
        if selected_status_8_2:
            st.write('MSE: {:.4f}'.format(mse))
        if selected_status_8_3:
            st.write('MAE: {:.4f}'.format(mae))
        st.write('')
    evaluate_prediction(df_test['Total_kWh'],df_test['Prediction'],'XGBoost')
    





