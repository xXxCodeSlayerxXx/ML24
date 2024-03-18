from collections import Counter
from keras.src.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import gc
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')
import tensorflow as tf
np.random.seed(84)
tf.random.set_seed(84)
tf.config.threading.set_intra_op_parallelism_threads(84)
tf.random.set_seed(84)

scaled_cols = ['GDP (current LCU) [NY.GDP.MKTP.CN]', 'Current health expenditure per capita (current US$) [SH.XPD.CHEX.PC.CD]',
               'Life expectancy at birth, total (years) [SP.DYN.LE00.IN]',
               'Urban population (% of total population) [SP.URB.TOTL.IN.ZS]',
               'Political Stability and Absence of Violence/Terrorism: Estimate [PV.EST]', 'Government Effectiveness: Estimate [GE.EST]',
               'Control of Corruption: Estimate [CC.EST]', 'Population density [EN.POP.DNST]']


def hyperparameter_tuning(params):
    # best_loss = np.inf
    # best_params = None
    # best_model = None
    total_loss = []
    n_splits = 3
    look_back, epochs, learning_rate, optimizer, neurons, dropout_rate, data = params

    # #Cross-validation
    for split in range(-n_splits,0):
        years = data['Time'].unique()

        #Last year is for validation
        train_data = data[data["Time"].isin(years[:split])]
        val_data = data[data["Time"].isin(years[split-look_back:split+1] if split != -1 else years[split-look_back:])] #Take two years from train years to make the prediction

        scaler = StandardScaler()
        # Fitting the scaler only on the training data
        scaler.fit(train_data[scaled_cols])

        train_data.loc[:, scaled_cols] = scaler.transform(train_data[scaled_cols])
        val_data.loc[:, scaled_cols] = scaler.transform(val_data[scaled_cols])

        model = build_model(learning_rate=learning_rate, num_time_steps=look_back+1,
                                                    num_features=len(scaled_cols), dropout_rate=dropout_rate,
                                                    optimizer=optimizer, neurons=neurons)
        
        model = fit_model(model, train_data, look_back=look_back, epochs=epochs, split="train")[0]
        val_loss = fit_model(model, val_data, look_back=look_back, epochs=1, split="val")[3]
        total_loss.append(val_loss)

    print(f"Trial completed with validation loss={val_loss}", flush=True)

    result = pd.DataFrame([{'look_back': look_back, 'epochs': epochs,
                            'learning_rate': learning_rate, 'optimizer': optimizer,
                            'neurons': neurons, 'dropout_rate': dropout_rate,
                            'validation_loss': np.mean(total_loss)}])
    return result

def fit_model(model, data, look_back = 2, epochs=1, split="train"):
    FourD_dataX, FourD_dataY = [], []
    countries = data['Country Name'].unique()
    countries_to_index = {}
    total_loss = 0
    losses = {}
    all_actuals = []
    all_predictions = []
    i=0
    for epoch in range(0, epochs):
        print(epoch, flush=True)
        np.random.shuffle(countries)
        for country in countries:
            if epoch == 0:
                X_country, y_country = create_dataset(data[data['Country Name'] == country].reset_index(drop=True),
                                                        look_back)
                FourD_dataX.append(X_country)
                FourD_dataY.append(y_country)

                countries_to_index[country] = i

                i += 1
            
            if epoch > 0:
                X_country = stacked_dataX[countries_to_index[country],:,:,:]
                y_country = stacked_dataY[countries_to_index[country],:]

            if len(X_country) == 0:
                print(f"Skipping country {country} due to insufficient data.")
                continue
            if split == "train":
                model.fit(X_country, y_country, verbose=0)
            if split == "val" or split == "test":
                country_predict = model.predict(X_country, verbose=0)
                total_loss += (y_country - country_predict)**2
                if split == "test":
                    losses[country] = (y_country - country_predict)**2
                    all_actuals.extend(y_country)
                    all_predictions.extend(country_predict.flatten())
        
        if epoch == 0:
            stacked_dataX = np.stack(np.array(FourD_dataX), axis=0)
            stacked_dataY = np.stack(np.array(FourD_dataY), axis=0)

    total_loss = total_loss/len(countries)

    return model, stacked_dataX, stacked_dataY, total_loss, losses, all_actuals, all_predictions

def build_model(learning_rate, num_time_steps, num_features, dropout_rate=0.2, optimizer='adam', neurons=32):

    # Select optimizer
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        opt = Adagrad(learning_rate=learning_rate)
    elif optimizer == 'adadelta':
        opt = Adadelta(learning_rate=learning_rate)
    else:
        opt = Adam(learning_rate=learning_rate)

    model = Sequential([
        LSTM(neurons, input_shape=(num_time_steps, num_features)),
        Dropout(dropout_rate),
        Dense(64, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer=opt, loss='mse')
    return model

def create_dataset(dataset, look_back = 2):
    dataX, dataY = [], []
    dataset = dataset.reset_index(drop=True)
    for i in range(len(dataset) - look_back):
        a = dataset.loc[i:(i + look_back), scaled_cols].values  # Ends at `look_back - 1` to get 5 steps
        dataX.append(a)
        # Assumes the target is right after the last step in `a`
        dataY.append(dataset.loc[i + look_back, 'Population growth (annual %) [SP.POP.GROW]'])

    return np.array(dataX), np.array(dataY)
