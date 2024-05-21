import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import cvxopt as opt
from cvxopt import blas, solvers
from cvxopt import matrix as opt_matrix, solvers



def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 10000
    
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    
    df = df[column_order]
   
    return df


def optimal_portfolio(portfolio_daily_ROR):
    N = 100 # This is the number of iterations
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)] # mus determines the range of expected portfolio_daily_ROR you are considering
    n = portfolio_daily_ROR.shape[1]
    S = opt.matrix(np.cov(portfolio_daily_ROR, rowvar=False))  # Covariance matrix
    pbar = opt.matrix(np.mean(portfolio_daily_ROR, axis=0))  # Expected portfolio_daily_ROR

    # Define constraints (e.g., individual asset weight limits)
    G = -opt.matrix(np.eye(n))  # Negative identity matrix for weights <= 1
    h = opt.matrix(0.0, (n, 1))  # All weights >= 0

    # Add the lower bound constraint for weights (>= 0)
    lb = opt_matrix(0.0, (n, 1))

    # Define constraint to ensure the sum of weights equals 1
    max_asset_weight = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b, options={'show_progress': False})['x']
                  for mu in mus]

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [opt.solvers.qp(opt.matrix(mu * S), -pbar, G, h, A, b, lb=lb)['x'] for mu in mus]

    # Calculate portfolio returns and risks
    optimal_portfolio_returns = [blas.dot(pbar, x) for x in portfolios]
    optimal_portfolio_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # Calculate the Sharpe ratio for each portfolio
    sharpe_ratios = [r / d for r, d in zip(optimal_portfolio_returns, optimal_portfolio_risks)]

    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))

    # Get the optimal portfolio weights
    optimal_weights = portfolios[max_sharpe_ratio_index]
    return optimal_weights, optimal_portfolio_returns, optimal_portfolio_risks

start_date = '2004-01-01'
end_date = '2024-01-01' 

tickers = ['AAPL', 'SAF.PA','ITX.MC', 'VIV.PA', 'CIE.MC']

stocks_data = yf.download(tickers, start=start_date, end=end_date)
print(stocks_data)

#selected = list(stocks_data.columns[1:])
rtos1 = stocks_data['Adj Close']
print(rtos1)
rtos = stocks_data['Adj Close'].pct_change()
rtos = rtos.fillna(0)
rtos= pd.DataFrame(rtos)
#print(rtos)

rtos_esperados = rtos.mean()
#print(rtos_esperados)
risk = rtos.cov()

#correlación
optimal_portfolio_by_returns = rtos.keys()
print(rtos[optimal_portfolio_by_returns].corr())


datetime = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
            2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
'''R_cartera = []
for i in range(len(rtos)):
    rendimiento_cartera = rtos.AAPL[i] + rtos['CIE.MC'][i] + rtos['ITX.MC'][i] + rtos.MSFT[i] + rtos['SAF.PA'][i]
    R_cartera.append(rendimiento_cartera)
print(R_cartera)
plt.plot(R_cartera)
#plt.show()'''

y = R_cartera1 = []
for i in range(len(rtos)):
    rendimiento_cartera = rtos1.AAPL[i] + rtos1['CIE.MC'][i] + rtos1['ITX.MC'][i] + rtos1['VIV.PA'][i] + rtos1['SAF.PA'][i]
    R_cartera1.append(rendimiento_cartera)
#print(R_cartera1)

#print(len(R_cartera1))
ax = plt.subplot()
plt.plot(R_cartera1)
plt.xticks([0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855, 4112, 4369, 4626, 4883, 5140])
ax.set_xticklabels(datetime)
plt.xlabel('Años')
plt.ylabel('Rendimiento acumulado')
plt.title('Rendimiento de la cartera a lo largo de los años')
plt.show()


random_portfolios = return_portfolios(rtos_esperados, risk)
random_portfolios.plot.scatter(x='Volatility', y='Returns',  color= '#41727e')
plt.xlabel('Volatilidad (Desviación Std)')
plt.ylabel('Rendimientos esperados')
plt.title('Combinaciones de carteras')
#plt.show()


weights, returns, risks = optimal_portfolio(rtos)
plt.plot(risks, returns, 'y-o', color = '#ed6068')
plt.show()


'''high_return = random_portfolios[random_portfolios['Returns']==random_portfolios['Returns'].max()]
#pd.set_option('display.max_columns', None)
print(high_return)'''
def optimal_portfolio(portfolio_daily_ROR):
    N = 2000 # This is the number of iterations
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)] # mus determines the range of expected portfolio_daily_ROR you are considering
    n = portfolio_daily_ROR.shape[1]
    S = opt.matrix(np.cov(portfolio_daily_ROR, rowvar=False))  # Covariance matrix
    pbar = opt.matrix(np.mean(portfolio_daily_ROR, axis=0))  # Expected portfolio_daily_ROR

    # Define constraints (e.g., individual asset weight limits)
    G = -opt.matrix(np.eye(n))  # Negative identity matrix for weights <= 1
    h = opt.matrix(0.0, (n, 1))  # All weights >= 0

    # Add the lower bound constraint for weights (>= 0)
    lb = opt_matrix(0.0, (n, 1))

    # Define constraint to ensure the sum of weights equals 1
    max_asset_weight = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b, options={'show_progress': False})['x']
                  for mu in mus]

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [opt.solvers.qp(opt.matrix(mu * S), -pbar, G, h, A, b, lb=lb)['x'] for mu in mus]

    # Calculate portfolio returns and risks
    optimal_portfolio_returns = [blas.dot(pbar, x) for x in portfolios]
    optimal_portfolio_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # Calculate the Sharpe ratio for each portfolio
    sharpe_ratios = [r / d for r, d in zip(optimal_portfolio_returns, optimal_portfolio_risks)]

    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))

    # Get the optimal portfolio weights
    optimal_weights = portfolios[max_sharpe_ratio_index]
    return optimal_weights, optimal_portfolio_returns, optimal_portfolio_risks

optimal_weights, returns, risks = optimal_portfolio(rtos)
print(optimal_weights)

'''#presupuesto = 10000
Appl_pres = 14391.93
Cie_pres = 1866.44
Itx_pres = 3429.21
Saf_pres = 4.62
Viv_pres = 1940.35

R_cartera_real = []

Inv = Appl_pres * rtos1.AAPL[1] + Cie_pres * rtos1['CIE.MC'][1] + Itx_pres * rtos1['ITX.MC'][1] + Saf_pres * rtos1['SAF.PA'][1] + Viv_pres * rtos1['VIV.PA'][1]
print(Inv)'''

