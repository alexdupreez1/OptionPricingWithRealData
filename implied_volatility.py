
from binomialTree import *
from parameter_estimations import *
import matplotlib.pyplot as plt




def compute_implied_volatility(tree, maturity, interest, strikeprice, historical_volatility, market_price,iterations):

    '''
    Description
    -----------
    Compute the implied volatility of the American put option using the Bisection method
    '''

    #Initial interval for bisection method
    lower_sigma_bound = 0
    upper_sigma_bound = 0.80

    theoretical_prices = [] #Store the computed theoretical price for each iteration

    for i in range(0,iterations):

        midpoint_sigma = (lower_sigma_bound + upper_sigma_bound)/2 #Compute midpoint of interval
 
        theoretical_price = AmericanPutOption(tree, maturity, interest, strikeprice, midpoint_sigma)[0][0]
        theoretical_prices.append(theoretical_price)

        if(theoretical_price > market_price):
            upper_sigma_bound = midpoint_sigma
        else:
            lower_sigma_bound = midpoint_sigma

    implied_volatility = midpoint_sigma #The final implied volatility is equal to the final midpoint
    
    return implied_volatility, theoretical_prices




    
if __name__ == "__main__":

    # real data of stock and put option
    S_0 = 74.09
    K = 85.00
    # Time to expiration
    weeks = 31 # weeks until expiry
    days = 5 # days in a regular trading week
    holidays = 5 # 19.02, 29.03, 27.05, 19.06, 04.07 (https://www.nyse.com/markets/hours-calendars)
    tradingdays = 252

    T = (weeks*days - holidays) / tradingdays
    # continuously compounded interest rate 

    R = 0.0522 # (https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2024)
    r = np.log(1 + R)

    stock_prices = caluclate_stock_returns()
    vol = estimate_volatility(stock_prices)
    sigma = np.sqrt(vol)

    tree = build_tree(S_0, sigma, T, 500)

    market_price = 14.70

    #theoretical_value = compute_implied_volatility(tree, T, r, K, sigma, market_price,100)

