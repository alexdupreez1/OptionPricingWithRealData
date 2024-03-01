import pandas as pd
import numpy as np

def caluclate_stock_returns():

    file_path = 'DAVA.csv'
    stock_data = pd.read_csv(file_path)
    
    stock_price_list = stock_data['Close'].tolist()

    stock_returns = []

    for i in range(1,len(stock_price_list)):

        stock_return = np.log(stock_price_list[i]/stock_price_list[i-1])
        stock_returns.append(stock_return)

    return stock_returns

def estimate_volatility(stock_returns):

    squared_return_difference = 0

    for i in range(1,len(stock_returns)):

        squared_return_difference += (stock_returns[i] - stock_returns[i-1])**2


    volatility = 252 * (1/(len(stock_returns)-1)) * squared_return_difference

    return volatility



if __name__ == '__main__':

    stock_returns = caluclate_stock_returns()

    volatility = estimate_volatility(stock_returns)
     







