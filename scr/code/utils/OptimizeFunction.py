import numpy as np
from scipy.optimize import minimize

def calc_sharpe_ratio_new(weights, returns):
    """
    Calculates the Sharpe ratio for a given set of portfolio weights and returns.

    Args:
        weights (numpy.ndarray): The weights of the assets in the portfolio. Shape (tickers,).
        returns (numpy.ndarray): The daily returns of the assets. Shape (days, tickers).

    Returns:
        float: The negative Sharpe ratio of the portfolio (negative because we minimize).
    """
    mean_y = np.mean(np.sum(returns, axis=1))
    center_y =np.sum(returns, axis=1)-mean_y
    cov_matrix = np.cov(center_y, rowvar=False)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    portfolio_return = np.sum(np.sum(returns,axis=1) * weights)
    sharpe_ratio = portfolio_return / portfolio_std
    return -sharpe_ratio  # Negative because we minimize

def mean_variance_optimization(weights, returns, risk_free_rate=0.0):
    """
    Performs Mean-Variance Optimization (MVO) to find the optimal portfolio weights.

    Args:
        weights (numpy.ndarray): The initial weights of the assets in the portfolio. Shape (tickers,).
        mean_returns (numpy.ndarray): The mean returns of the assets. Shape (tickers,).
        cov_matrix (numpy.ndarray): The covariance matrix of the asset returns. Shape (tickers, tickers).
        risk_free_rate (float): The risk-free rate. Default is 0.0.

    Returns:
        float: The negative Sharpe ratio of the portfolio (negative because we minimize).
    """
    returns = [[sum(row) for row in two_d] for two_d in returns]
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(mean_returns, rowvar=False)
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
    return sharpe_ratio  # Negative because we minimize


def global_minimum_variance_portfolio(weights, returns):
    """
    Optimizes the portfolio to achieve the Global Minimum Variance Portfolio (GMVP).

    Args:
        returns (numpy.ndarray): The daily returns of the assets. Shape (days, tickers).

    Returns:
        numpy.ndarray: The optimized weights for the GMVP.
    """
    # Calculate the covariance matrix of the returns
    returns = [[sum(row) for row in two_d] for two_d in returns]
    # weights = weights[:, np.newaxis]
    cov_matrix = np.cov(returns, rowvar=False)
    return np.dot(weights.T, np.dot(cov_matrix, weights))


def most_diversified_portfolio(weights, returns):
    """
    Optimizes the portfolio to achieve the Most Diversified Portfolio (MDP).

    Args:
        returns (numpy.ndarray): The daily returns of the assets. Shape (days, tickers).

    Returns:
        numpy.ndarray: The optimized weights for the MDP.
    """
    returns = [[sum(row) for row in two_d] for two_d in returns]
    # Calculate the covariance matrix of the returns
    volatilities = np.std(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)

    # Objective function: negative diversification ratio
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    weighted_volatilities = np.dot(weights, volatilities)
    return -weighted_volatilities / portfolio_volatility


def equal_weighted_portfolio(returns):
    """
    Assigns equal weights to each asset in the portfolio.

    Args:
        returns (numpy.ndarray): The daily returns of the assets. Shape (days, tickers).

    Returns:
        numpy.ndarray: The equal weights for the portfolio.
    """
    num_assets = returns.shape[1]
    equal_weights = np.ones(num_assets) / num_assets
    return equal_weights


def inverse_volatility_portfolio(returns):
    """
    Assigns weights to each asset in the portfolio based on the inverse of their volatilities.

    Args:
        returns (numpy.ndarray): The daily returns of the assets. Shape (days, tickers).

    Returns:
        numpy.ndarray: The weights for the portfolio based on inverse volatilities.
    """

    # Calculate the standard deviation (volatility) of each asset
    try:
        # Calculate the standard deviation (volatility) of each asset
        returns = [[sum(row) for row in two_d] for two_d in returns]
        volatilities = np.std(returns, axis=0)

        # Calculate the inverse of the volatilities
        inverse_volatilities = 1 / volatilities

        # Normalize the inverse volatilities to get the portfolio weights
        weights = inverse_volatilities / np.sum(inverse_volatilities)

        return weights
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.array([])  # Return an empty array or a default value


def optimizeFunction(method, returns, top_n, risk_free_rate=0.05):
    """
    Optimizes the portfolio based on the specified method.

    Args:
        method (str): The optimization method to use. Options are 'SRO', 'MVO', 'GMVP', 'MDP', 'EWP', 'IVP'.
        returns (numpy.ndarray): The daily returns of the assets. Shape (days, tickers).
        top_n (int): The number of top assets to include in the portfolio.
        risk_free_rate (float): The risk-free rate to use in the optimization. Default is 0.05.

    Returns:
        numpy.ndarray: The optimized weights for the portfolio.
    """
    initial_weights = np.ones(top_n) / top_n
    bounds = [(0, 1) for _ in range(top_n)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    #Sharpe Ratio Optimization
    if method == 'SRO':
        result = minimize(calc_sharpe_ratio_new, initial_weights, args=(returns,), bounds=bounds,
                          constraints=constraints)  # Negative because we minimize
        weights = result.x
    #Mean-Variance Optimization (MVO)
    elif method == 'MVO':
        result = minimize(mean_variance_optimization, initial_weights, args=(returns, risk_free_rate,), method='SLSQP',
                          bounds=bounds, constraints=constraints)  # Negative because we minimize
        weights = result.x
    #Global Minimum Variance Portfolio (GMVP)
    elif method == 'GMVP':
        result = minimize(global_minimum_variance_portfolio, initial_weights, args=(returns,), method='SLSQP',
                          bounds=bounds, constraints=constraints)  # Return an empty array or a default value
        weights = result.x
    #Most Diversified Portfolio (MDP)
    elif method == 'MDP':
        result = minimize(most_diversified_portfolio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds,
                          constraints=constraints)  # Return an empty array or a default value
        weights = result.x
    #Equal-Weighted Portfolio (EWP)
    elif method == 'EWP':
        return equal_weighted_portfolio(returns)  # Return an empty array or a default value
    #Inverse-Volatility Portfolio (IVP)
    elif method == 'IVP':
        return inverse_volatility_portfolio(returns)  # Return an empty array or a default value
    return weights
