import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import cvxopt as opt
from cvxopt import blas, solvers


def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.values)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

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

x = datetime = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
            2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

start_date = '2004-01-01'
end_date = '2024-01-01' 

tickers = ['AAPL', 'SAF.PA','ITX.MC', 'VIV.PA', 'CIE.MC']


##############################################
### RENDIMIENTO POR PONDERACIONES ÓPTIMAS:###
#############################################

AAPL = yf.download('AAPL', start=start_date, end=end_date)
SAF = yf.download('SAF.PA', start=start_date, end=end_date)
VIV = yf.download('VIV.PA', start=start_date, end=end_date)  
ITX = yf.download('ITX.MC', start=start_date, end=end_date)
CIE = yf.download('CIE.MC', start=start_date, end=end_date)

df = pd.DataFrame({'AAPL': AAPL['Adj Close'], 'SAF': SAF['Adj Close'], 'VIV': VIV['Adj Close'], 'ITX': ITX['Adj Close'], 'CIE': CIE['Adj Close']})
#print(df)

'''date_column = df.index.strftime('%Y-%m-%d')
print(date_column)'''

rtos = df.pct_change()
rtos = rtos.fillna(0)
#print(rtos)

Appl_por = 0.463
Cie_por = 0.175
Itx_por = 0.227
Saf_por = 0.0056
Viv_por = 0.129 

R_cartera = []
for i in range(len(df)):
    rendimiento_cartera = df.AAPL[i] * Appl_por + df.CIE[i] * Cie_por + df.ITX[i] * Itx_por + df.VIV[i] * Viv_por + df.SAF[i] * Saf_por
    R_cartera.append(rendimiento_cartera)
#print(R_cartera)


df['rendimiento total cartera'] = R_cartera
#print(df)

df = df.drop('AAPL', axis=1)
df = df.drop('CIE', axis=1) 
df = df.drop('ITX', axis=1) 
df = df.drop('VIV', axis=1)    
df = df.drop('SAF', axis=1)
#print(df)
df=df.dropna()
#df.to_csv('rend_real.csv')


fig, ax = plt.subplots(figsize=(12,7))
plt.plot(R_cartera,  color= '#41727e')
plt.xticks([0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855, 4112, 4369, 4626, 4883, 5140])
ax.set_xticklabels(datetime)
plt.xlabel('Años')
plt.ylabel('Rendimiento acumulado')
plt.title('Rendimiento de la cartera a lo largo de los años')
plt.show()

###################################################
### RENDIMIENTO POR PONDERACIONES IGUALITARIAS:###
#################################################

AAPL = yf.download('AAPL', start=start_date, end=end_date)
SAF = yf.download('SAF.PA', start=start_date, end=end_date)
VIV = yf.download('VIV.PA', start=start_date, end=end_date)  
ITX = yf.download('ITX.MC', start=start_date, end=end_date)
CIE = yf.download('CIE.MC', start=start_date, end=end_date)

df = pd.DataFrame({'AAPL': AAPL['Adj Close'], 'SAF': SAF['Adj Close'], 'VIV': VIV['Adj Close'], 'ITX': ITX['Adj Close'], 'CIE': CIE['Adj Close']})
#print(df)

rtos = df.pct_change()
rtos = rtos.fillna(0)
#print(rtos)


R_cartera = []
for i in range(len(df)):
    rendimiento_cartera = df.AAPL[i] + df.CIE[i] + df.ITX[i] + df.VIV[i] + df.SAF[i]
    R_cartera.append(rendimiento_cartera)
#print(R_cartera)

df['rendimiento total cartera'] = R_cartera
#print(df)

df = df.drop('AAPL', axis=1)
df = df.drop('CIE', axis=1) 
df = df.drop('ITX', axis=1) 
df = df.drop('VIV', axis=1)    
df = df.drop('SAF', axis=1)
#print(df)
df=df.dropna()
#df.to_csv('cart3ra.csv')

ax = plt.subplot()
plt.plot(R_cartera, color= '#41727e')
plt.xticks([0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855, 4112, 4369, 4626, 4883, 5140])
ax.set_xticklabels(datetime)
plt.xlabel('Años')
plt.ylabel('Rendimiento acumulado')
plt.title('Rendimiento de la cartera a lo largo de los años')
plt.show()


###########################################
### RENDIMIENTO CON PRESUPUESTO INICAL:###
#########################################

AAPL = yf.download('AAPL', start=start_date, end=end_date)
SAF = yf.download('SAF.PA', start=start_date, end=end_date)
VIV = yf.download('VIV.PA', start=start_date, end=end_date)  
ITX = yf.download('ITX.MC', start=start_date, end=end_date)
CIE = yf.download('CIE.MC', start=start_date, end=end_date)

df = pd.DataFrame({'AAPL': AAPL['Adj Close'], 'SAF': SAF['Adj Close'], 'VIV': VIV['Adj Close'], 'ITX': ITX['Adj Close'], 'CIE': CIE['Adj Close']})
#print(df)

rtos = df.pct_change()
rtos = rtos.fillna(0)
#print(rtos)

Appl_pres = 14391.93
Cie_pres = 1866.44
Itx_pres = 3429.21
Saf_pres = 4.62
Viv_pres = 1940.35

R_cartera = []
for i in range(len(df)):
    rendimiento_cartera = df.AAPL[i] * Appl_pres + df.CIE[i] * Cie_pres + df.ITX[i] * Itx_pres + df.VIV[i] * Viv_pres + df.SAF[i] * Saf_pres
    R_cartera.append(rendimiento_cartera)
#print(R_cartera)

df['rendimiento total cartera'] = R_cartera
#print(df)

df = df.drop('AAPL', axis=1)
df = df.drop('CIE', axis=1) 
df = df.drop('ITX', axis=1) 
df = df.drop('VIV', axis=1)    
df = df.drop('SAF', axis=1)
#print(df)
df=df.dropna()
#df.to_csv('rend_real_total.csv')


ax = plt.subplot()
plt.plot(R_cartera, color= '#41727e')
plt.xticks([0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855, 4112, 4369, 4626, 4883, 5140])
ax.set_xticklabels(datetime)
plt.xlabel('Años')
plt.ylabel('Rendimiento acumulado')
plt.title('Rendimiento de la cartera a lo largo de los años')
plt.show()


