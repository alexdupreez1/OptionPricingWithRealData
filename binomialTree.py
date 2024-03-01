import numpy as np
from scipy.stats import norm

def build_tree(stockprice, volatility, maturity, nsteps):
    # up and down factor 
    u = np.exp(volatility * np.sqrt(maturity/nsteps))
    d = np.exp(-volatility * np.sqrt(maturity/nsteps))
    
    # initialitze matrix
    matrix = np.zeros((nsteps + 1, nsteps + 1))
    
    # iterate through rows to set up potential stock prices
    for row_index in range(nsteps+1):
        for column_index in range(row_index+1):
            
            # change matrix entries depending on up and down moves at the given time step
            matrix[row_index, column_index] = stockprice * u**column_index * d**(row_index - column_index)
            
    # return result matrix       
    return matrix

def EuropeanPutOption(tree, maturity, interest, strikeprice, volatility):
    
    # determine option matrix size
    row = tree.shape[0]
    column = tree.shape[1]
    
    # initalize time steps
    dt = maturity / (row-1)
    
    # set required parameters
    u = np.exp(volatility * np.sqrt(maturity/(row-1)))
    d = np.exp(-volatility * np.sqrt(maturity/(row-1)))
    p = (np.exp(interest*dt) - d) / (u - d)
    
    # initialize option price matrix 
    option_price = np.zeros((row, column))
    
    # calculate option prices for maturity date
    for column_index in range(column):
        option_price[row-1, column_index] = max(0, strikeprice - tree[row-1, column_index])
    
    # back-track option prices for previous dates
    for i in range(row-2, -1, -1): 
        for j in range(i+1): 
            
            up = option_price[i + 1, j + 1]
            down = option_price[i + 1, j] 
            
            option_price[i, j] = np.exp(-interest * dt) * (p * up + (1-p) * down)
            
    # return matrix of option prices
    return option_price

def ExactSampling(stockprice, maturity, interest, volatility, nsteps):
    # initialize step size and initial stockprice S0
    delta_t = maturity/nsteps
    St = [stockprice]
    
    # iterate through time steps to generate stock prices with geometric Brownian motion
    for m in range(1,nsteps + 1):
        Zm = np.random.normal(0,1)
        currentStockprice = St[-1] * np.exp( (interest - (volatility**2)/2) * delta_t + volatility * np.sqrt(delta_t) * Zm )
        St.append(currentStockprice)
    
    return np.array(St)

def BlackScholesPut(stockprice, maturity, interest, volatility, strikeprice, nsteps):
    delta_t = maturity/nsteps # time step size
    S = ExactSampling(stockprice, maturity, interest, volatility, nsteps) # array of stockprices
    
    BlackScholesPutValue = []
    
    for time in range(0, nsteps):
        tau = maturity - time * delta_t
        d1 = 1/(volatility*np.sqrt(tau)) * ( np.log(S[time]/strikeprice) + (interest + (volatility**2)/2) * tau )
        d2 = d1 - volatility*np.sqrt(tau)
        
        Vt = strikeprice * np.exp(-interest * tau) * norm.cdf(-d2) - S[time] * norm.cdf(-d1)
        BlackScholesPutValue.append(Vt)
    
    BlackScholesPutValue.append(max(strikeprice-S[-1], 0))
    
    return np.array(BlackScholesPutValue)[0]

# Put (note the tree of stock prices is the same for European and American options)
def AmericanPutOption(tree, maturity, interest, strikeprice, volatility):
    
    # determine option matrix size
    row = tree.shape[0]
    column = tree.shape[1]
    
    # initalize time steps
    dt = maturity / (row-1)
    
    # set required parameters
    u = np.exp(volatility * np.sqrt(maturity/(row-1)))
    d = np.exp(-volatility * np.sqrt(maturity/(row-1)))
    p = (np.exp(interest*dt) - d) / (u - d)
    
    # initialize option price matrix 
    option_price = np.zeros((row, column))
    
    # calculate option prices for maturity date
    for column_index in range(column):
        option_price[row-1, column_index] = max(0, strikeprice - tree[row-1, column_index])
    
    # back-track option prices for previous dates
    for i in range(row-2, -1, -1): 
        for j in range(i+1): 
            
            up = option_price[i + 1, j + 1]
            down = option_price[i + 1, j] 
            
            option_price[i, j] = max(strikeprice-tree[i,j], np.exp(-interest * dt) * (p * up + (1-p) * down))
            
    # return matrix of option prices
    return option_price