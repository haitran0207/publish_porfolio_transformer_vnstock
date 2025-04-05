import pandas as pd
import json as js
import numpy as np

def prepair_data_new(path, window_x, window_y,step_roll=1):
    df = pd.read_csv(path)
    df = df.ffill()

    df['date'] = df.date.apply(pd.Timestamp)
    df = df.pivot_table(index='date', columns='symbol')
    df[np.isnan(df)] = 0

    if 'VNINDEX' in df.columns.get_level_values('symbol'):
        vnindex_df = df.xs('VNINDEX', level='symbol', axis=1, drop_level=False)

        df = df.drop(columns=['VNINDEX'], level='symbol')

        vnindex_df = vnindex_df.close
        daily_return_vnindex = ((vnindex_df.shift(-1) - vnindex_df) / vnindex_df).shift(1)
        daily_return_vnindex = daily_return_vnindex.interpolate(method='linear', limit_area="inside",
                                                                limit_direction='both', axis=0)
        y_vnindex = daily_return_vnindex.values
        y_vnindex[np.isnan(y_vnindex)] = 0
        y_vnindex = rolling_array(y_vnindex[window_x:], stepsize=1, window=window_y, step_roll=step_roll)
    else:
        y_vnindex = None

    # Select tickers not NaN in final day
    columns = df.close.columns[~df.close.iloc[-1].isna()]
    df = df.iloc[:, df.columns.get_level_values(1).isin(columns)]

    # Interpolate missing values for all columns
    for col in df.columns.levels[0]:
        df[col] = df[col].interpolate(method='linear', limit_area='inside', limit_direction='both', axis=0)

    # Calculate daily return based on close
    close = df.close
    daily_return = ((close.shift(-1) - close) / close).shift(1)
    daily_return = daily_return.interpolate(method='linear', limit_area="inside", limit_direction='both', axis=0)

    # Get tickers
    tickers = df.close.columns

    # Reshape DataFrame to have all columns in X
    X = df.values.reshape(df.shape[0], len(df.columns.levels[0]), -1)
    y = daily_return.values

    dates = df.index.values
    y[np.isnan(y)] = 0

    # Create rolling arrays
    X = rolling_array(X[:-window_y], stepsize=1, window=window_x, step_roll=step_roll)
    X_return = rolling_array(y[:-window_y], stepsize=1, window=window_x, step_roll=step_roll)
    y = rolling_array(y[window_x:], stepsize=1, window=window_y, step_roll=step_roll)
    dates = rolling_array(dates[window_x:], stepsize=1, window=window_y, step_roll=step_roll)

    X = np.moveaxis(X, -1, 1)
    X_return = np.moveaxis(X_return, -1, 1)
    y = np.swapaxes(y, 1, 2)

    return X, y, y_vnindex, tickers, dates, X_return, close

def rolling_array(a, stepsize=1, window=60, step_roll=1):
    n = a.shape[0]
    t = np.array(list(a[i:i + window:stepsize] for i in range(0, n - window + 1,step_roll)))
    return np.stack(t)


def load_config_file(path):
    with open(path, 'r') as file:
        f = js.load(file)
    return f


class MyEncoder(js.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)