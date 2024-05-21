import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import cvxopt as opt
from cvxopt import blas, solvers

start_date = '2004-01-01'
end_date = '2024-01-01' 

tickers = 'ES=F'
SP500 = yf.download(tickers, start=start_date, end=end_date)
print(SP500)

rtos1 = SP500['Adj Close']

y = R_cartera1 = []
for i in range(len(rtos1)):
    SP_500 = rtos1[i]
    R_cartera1.append(SP_500)


datetime = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 
            2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]


ax = plt.subplot()
plt.plot(R_cartera1, color= '#41727e')
plt.xticks([0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855, 4112, 4369, 4626, 4883, 5140])
ax.set_xticklabels(datetime)
plt.xlabel('Años')
plt.ylabel('Rendimiento acumulado')
plt.title('Rendimiento real del S&P500 a lo largo de los años')
plt.show()


df = pd.DataFrame({'SP500':SP500['Adj Close']})
df.to_csv('SP500r.csv')

print(df)

#df.to_csv('SP500.csv')
print(len(df))
'''
#precio inicial = 1109
SP500_pres = 9.017

SP = []

test = []
for i in range(len(rtos1)):
    if i <= 5044:
        SP500_1 = (df.SP500[i+1] - df.SP500[i]) 

        SP.append(SP500_1)
        gg = SP[i]
        test.append(gg)

s = 0
ds=[]
for i in range(4980):
    s += test[i]
    ds.append(s)
print(ds)

ax = plt.subplot()   
plt.plot(ds)
plt.xticks([0, 257, 514, 771, 1028, 1285, 1542, 1799, 2056, 2313, 2570, 2827, 3084, 3341, 3598, 3855, 4112, 4369, 4626, 4883, 5140])
ax.set_xticklabels(datetime)
plt.xlabel('Años')
plt.ylabel('Rendimiento real acumulado')
plt.title('Rendimiento real del S&P500 a lo largo de los años')
plt.show()

ds = pd.DataFrame(ds)
ds.to_csv('SP500real.csv') '''