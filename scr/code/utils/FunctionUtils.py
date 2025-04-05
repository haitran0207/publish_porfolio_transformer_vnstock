from tensorflow.keras import backend as k
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json

from scr.code.preData.preprocess_data import MyEncoder

tf.config.run_functions_eagerly(True)

def sharpe_ratio_loss(y_true, y_pred):
    epsilon = 1e-2
    y_pred_reshape = k.expand_dims(y_pred, axis=-1)
    z = y_true * y_pred_reshape
    z = k.sum(z, axis=1)
    sharpeRatio = k.mean(z, axis=1) / k.maximum(k.std(z, axis=1), epsilon)
    return -k.mean(sharpeRatio)


def sharpe_ratio(y_true, y_pred):
    epsilon = 1e-2

    y_pred_reshape = k.expand_dims(y_pred, axis=-1)
    z = y_true * y_pred_reshape
    z = k.sum(z, axis=1)

    return k.mean(z, axis=1) / k.maximum(k.std(z, axis=1), epsilon)



def sharpe_ratio_manual(y_true, y_pred, y_vnindex = None):
    epsilon = 1e-6
    top_n = 30
    y_true = tf.expand_dims(y_true, axis=0)
    y_pred = tf.expand_dims(y_pred, axis=0)
    # Get top n tickers
    indicates = tf.nn.top_k(y_pred, k=top_n).indices
    indicates = tf.expand_dims(indicates, axis=0)
    # Gather top n tickers
    y_true_gathered = tf.gather(y_true, indicates, batch_dims=1)

    # Portfolio standard deviation
    weights = np.full(top_n, 1 / top_n)
    mean_y_true = k.mean(k.sum(y_true_gathered[0][0], axis=1), axis=0)
    centered_y_true = k.sum(y_true_gathered[0][0], axis=1) - mean_y_true
    cov_matrix = np.dot(centered_y_true, tf.transpose(centered_y_true)) / tf.cast(tf.shape(centered_y_true)[0] - 1,tf.float64)
    portfolio_std = np.sqrt(np.dot(tf.transpose(weights), np.dot(cov_matrix, weights)))

    # Sharpe ratio
    if y_vnindex is not None:
        y_vnindex = tf.expand_dims(y_vnindex, axis=0)
        sharpeRatio_gathered = (k.sum(k.sum(y_true_gathered * 1/top_n, axis=2)[0]) -k.sum(y_vnindex))/k.maximum(portfolio_std, epsilon)
    else:
        sharpeRatio_gathered = k.sum(k.sum(y_true_gathered * 1 / top_n, axis=2)[0] - 0.05 / 240) / k.maximum(portfolio_std, epsilon)
    return sharpeRatio_gathered



def visualize_log(path_folder, model_name):
    """
    Visualizes the training and validation Sharpe ratio logs for the last 6 files in the specified folder.

    Args:
        path_folder (str): The path to the folder containing the log files.
        model_name (str): The name of the model whose logs are to be visualized.

    Returns:
        None
    """
    n_cols = 6
    n_rows = 1
    fig, axes = plt.subplots(ncols=n_cols, figsize=(20, 3))
    path_files = [os.path.join(path_folder, model_name, file) for file in
                  os.listdir(os.path.join(path_folder, model_name)) if
                  os.path.isfile(os.path.join(path_folder, model_name, file))]
    for i, path in enumerate(path_files[-6:]):
        with open(path) as f:
            history = json.loads(f.read())

        axes[i].plot(history['sharpe_ratio'][50:])
        axes[i].set_ylabel('sharpe_ratio')
        axes[i].set_xlabel('Epoch')
        axes[i].legend(['Train', 'Test'], loc='upper left')
    new_path = os.path.join('/'.join(path_folder.split('/')[:-1]), 'plot', model_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    ver = list(map(lambda x: int(x.split('.')[0]),
                   [file for file in os.listdir(new_path) if file.endswith('.png')]))
    if len(ver) > 0:
        ver = np.max(ver) + 1
    else:
        ver = 0
    plt.savefig(os.path.join(new_path, str(ver) + '.png'))


def save_model(model, path, model_name, optimizer=None):
    if optimizer is not None:
        ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(path, model_name), max_to_keep=5)
        ckpt_manager.save()
        print('Latest checkpoint saved!!')
    else:
        if not os.path.exists(os.path.join(path, model_name)):
            os.makedirs(os.path.join(path, model_name))
        ver = list(map(lambda x: int(x.split('.')[0]),
                       [file for file in os.listdir(os.path.join(path, model_name)) if file.endswith('.h5')]))
        if len(ver) > 0:
            ver = np.max(ver) + 1
        else:
            ver = 0
        model.save(os.path.join(path, model_name, str(ver) + '.h5'))
        print("Model saved at %s" % os.path.join(path, model_name))

def calc_sharpe_ratio_portfolio(weights, returns, window_y):
    """
    Calculates the Sharpe ratio for a given set of weights and daily returns for a portfolio.

    Args:
        weights (numpy.ndarray): The weights of the assets in the portfolio. Shape (tickers,).
        returns (numpy.ndarray): The daily returns of the assets. Shape (tickers, days).
        window_y (int): The number of trading days in a year (e.g., 252 for daily returns).

    Returns:
        float: The calculated Sharpe ratio.
    """
    epsilon = 1e-6

    # Calculate the portfolio returns
    portfolio_returns = np.sum(weights * np.sum(returns,axis=1))

    mean_y = np.mean(np.sum(returns, axis=1))
    center_y = np.sum(returns, axis=1) - mean_y
    cov_matrix = np.cov(center_y, rowvar=False)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


    # Calculate and return the Sharpe ratio
    sharpe_ratio = portfolio_returns / portfolio_std
    return sharpe_ratio, portfolio_returns, portfolio_std


def write_log(history, path_dir, name_file):
    """
    Writes the training history log to a specified directory and file.

    Args:
        history (History): The training history object containing the training metrics.
        path_dir (str): The directory path where the log file will be saved.
        name_file (str): The name of the log file.

    Returns:
        None
    """
    his = history.history if hasattr(history, 'history') else history
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(os.path.join(path_dir, name_file), 'w') as outfile:
        json.dump(his, outfile, cls=MyEncoder, indent=2)
    print("write file log at %s" % (os.path.join(path_dir, name_file)))

def save_optimized_weights_log(tickers_list,optimized_weights, start_dates, log_dir, model_name, formatted_time, sharpe_ratios, mean_returns, std_returns, portfolio_values, shares_helds,test_index):
    # Ensure the directory exists
    if len(optimized_weights) > 0:
        result_path = os.path.join(log_dir, model_name)
        os.makedirs(result_path, exist_ok=True)
        log_data = []
        for i in range(len(optimized_weights)):
            log_entry = {
                'start_date': start_dates[i].astype(str),
                'sharpe_ratio': round(sharpe_ratios[i], 2),
                'mean_return': round(mean_returns[i], 4),
                'std_return': std_returns[i],
                'portfolio_values': portfolio_values[i],
                'shares_helds': shares_helds[i],
                'tickers_list': tickers_list[i],
                'optimized_weights': [round(weight * 100, 2) for weight in optimized_weights[i]]
            }
            log_data.append(log_entry)

        result_path = os.path.join(result_path, f"weights_{test_index[-1]}_{formatted_time}.json")
        with open(result_path, 'w') as log_file:
            json.dump(log_data, log_file, indent=4)
        print(f"Optimized weights and start dates saved to {result_path}")