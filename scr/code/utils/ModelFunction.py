import gc

from sklearn.model_selection import TimeSeriesSplit

from scr.code.constants.constant import *
from scr.code.utils.FunctionUtils import *

import numpy as np
from datetime import datetime, timedelta

def calc_portfolio_new(results, y_val, date_val, tickers, window_y, top_n, optimize_method, risk_free_rate, model_name, his = None, X_ret=None, close=None, initial_capital = None,test_index = None):
    if initial_capital is None:
        initial_capital = 1000000
    results[results < 0.01] = 0
    optimized_weights = []
    tickers_portfolios = []
    sharpe_ratios = []
    mean_returns = []
    std_returns = []
    portfolio_values = []
    shares_helds = []
    current_portfolio_value = initial_capital
    shares_held = {}
    start_date = [date[0] for date in date_val]

    for i in range(len(results)):
        indices = [index for index, value in enumerate(results[i]) if value != 0]

        if i > 0:
            # Sell all stocks at the price of the restructuring date
            previous_date = start_date[i]
            current_portfolio_value = 0
            for ticker in shares_held:
                price = close[ticker].loc[previous_date]
                next_date = previous_date
                while np.isnan(price):
                    next_date += timedelta(days=1)
                    price = close[ticker].loc[next_date]
                current_portfolio_value += shares_held[ticker] * price
            shares_held = {}  # Reset shares held after selling

        returns = y_val[i][indices]
        tickers_list = [tickers[j] for j in indices]
        tickers_portfolios.append(tickers_list)
        weights_list = results[i][indices]
        optimized_weights.append(weights_list)
        sharpe_ratio, mean_return, std_return = calc_sharpe_ratio_portfolio(weights_list, returns, window_y)
        sharpe_ratios.append(sharpe_ratio)
        mean_returns.append(mean_return)
        std_returns.append(std_return)
        print('Start Buy Date: %s Sample %d : [ %s ]' % (
            start_date[i], i, ' '.join([f"{tickers_list[j]}: {weights_list[j] * 100:.2f}" for j in range(len(weights_list))])))
        print('Sharpe ratio of this portfolio: %s Mean return %s, Std return %s' % (str(sharpe_ratio), str(mean_return), str(std_return)))

        # Convert start_date[i] to a datetime object
        next_date = start_date[i]

        # Calculate number of shares bought
        for j in range(len(weights_list)):
            ticker = tickers_list[j]
            price = close[ticker].loc[start_date[i]]
            while price == 0:
                next_date += timedelta(days=1)
                if next_date.strftime('%Y-%m-%d') not in close[ticker].index:
                    shares_held[ticker] = 0
                    break
                price = close[ticker].loc[next_date.strftime('%Y-%m-%d')]
            else:
                shares_held[ticker] = (weights_list[j] * current_portfolio_value) / price

        # Calculate portfolio value
        portfolio_value = 0
        for ticker in shares_held:
            price = close[ticker].loc[start_date[i]]
            next_date = start_date[i]
            while np.isnan(price):
                next_date += timedelta(days=1)
                price = close[ticker].loc[next_date]
            portfolio_value += shares_held[ticker] * price

        shares_helds.append(shares_held)
        portfolio_values.append(portfolio_value)
        print('Portfolio value: %s' % str(portfolio_value))

    # Get current time
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
    if his is not None:
        write_log(his, './logs/%s' % model_name, "log_%d.txt" % (test_index[-1]))

    save_optimized_weights_log(tickers_portfolios, optimized_weights, start_date, 'results', model_name, formatted_time, sharpe_ratios, mean_returns, std_returns, portfolio_values, shares_helds,test_index)

    return portfolio_values[-1]

def packageData(X, X_return, dates, window_y, y):
    tscv = TimeSeriesSplit(n_splits=10)
    X_train = np.empty((0, X.shape[1], X.shape[2], X.shape[3]))
    y_train = np.empty((0, y.shape[1], y.shape[2]))
    X_ret = np.empty((0, X_return.shape[1], X_return.shape[2]))
    X_vali = np.empty((0, X.shape[1], X.shape[2], X.shape[3]))
    y_vali = np.empty((0, y.shape[1], y.shape[2]))
    date_vali = np.empty((0, dates.shape[1]), dtype=str)
    for train_index, test_index in tscv.split(X):
        # Split the data into training and validation sets
        X_tr, X_val, X_r = (X[train_index], X[test_index[range(window_y - 1, len(test_index), window_y)]],X_return[test_index[range(window_y - 1, len(test_index), window_y)]])
        y_tr, y_val, date_val = y[train_index], y[test_index[range(window_y - 1, len(test_index), window_y)]], dates[test_index[range(window_y - 1, len(test_index), window_y)]]

        X_train = np.concatenate((X_train, X_tr), axis=0)
        y_train = np.concatenate((y_train, y_tr), axis=0)
        X_ret = np.concatenate((X_ret, X_r), axis=0)
        X_vali = np.concatenate((X_vali, X_val), axis=0)
        y_vali = np.concatenate((y_vali, y_val), axis=0)
        date_vali = np.concatenate((date_vali, date_val.astype(str)), axis=0)
    return X_ret, X_train, X_vali, date_vali, y_train, y_vali

def TrainModel_New(optimize_method, model_name, close, tickers, model, X, X_return, dates, window_y, y, optimizer):
    tscv = TimeSeriesSplit(n_splits = n_fold)
    innitial_capital = 1000000
    for train_index, test_index in tscv.split(X):
        # Calculate test_index
        step = round(window_y/step_roll,0) # 5
        start_index = test_index[0]
        filtered_testindex = [index for index in test_index if (index - start_index) % step == 0]
        # Split the data into training and validation sets
        X_tr, X_val, X_r_tr, X_r_val = X[train_index], X[filtered_testindex], X_return[train_index] , X_return[filtered_testindex]
        y_tr, y_val, date_val = y[train_index], y[filtered_testindex], dates[filtered_testindex]

        # TrainModel
        if 'TRANSFORMER' in model_name:
            his= None
            his = model.fit([X_tr, X_r_tr], y_tr, batch_size=int(np.round(2000 / window_x, 0)), epochs=epochs)
            # Predict
            results = model.predict([X_val, X_r_val])
        else:
            his = model.fit(X_tr, y_tr, batch_size=int(np.round(4000 / window_x, 0)), epochs=epochs)
            results = model.predict(X_val)

        # Calculate Portfolio
        innitial_capital = calc_portfolio_new(results, y_val, date_val, tickers, window_y, top_n, optimize_method, risk_free_rate, model_name, his, X_r_tr, close, innitial_capital,test_index)

        # Save Model
    save_model(model, pretrain_model_path, model_name, optimizer)

    # visualize_log(logs_path, model_name)
    gc.collect()