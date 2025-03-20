import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from statsmodels.tsa.stattools import acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', round(stats.skew(data), 3))
    print('Kurtosis:', round(stats.kurtosis(data), 3))
    print('Shapiro-Wilk p = ', round(100*stats.shapiro(data)[1], 1), '%')
    print('Jarque-Bera p = ', round(100*stats.jarque_bera(data)[1], 1), '%')
    print('Autocorrelation function analysis for ' + label)
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), label, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), label, '\n')

df = pd.read_excel("data.xlsx", sheet_name = 'data')
vol = df["Volatility"].values[1:]
N = len(vol)
print('Data size = ', N)
price = df['Price'].values
dividend = df['Dividends'].values
earnings = df['Earnings'].values
cpi = df['CPI'].values
inflation = np.diff(np.log(cpi))
nearngr = np.diff(np.log(earnings))
rearngr = nearngr - inflation
EY = earnings/price
plt.plot(range(1927, 2025), EY)
plt.xlabel('Years')
plt.title('Earnings Yield')
plt.savefig('EY.png')
plt.close()
nPrice = np.diff(np.log(price))
rPrice = nPrice - inflation
nTotal = np.array([np.log(price[k+1] + dividend[k+1]) - np.log(price[k]) for k in range(N)])
rTotal = nTotal - inflation
spreads = {}
spreads['BAA-AAA'] = (df['BAA'] - df['AAA']).values
spreads['AAA-Long'] = (df['AAA'] - df['Long']).values
spreads['Long-Short'] = (df['Long'] - df['Short']).values

DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'EY' : EY[:-1]})
for key in spreads:
    DFreg[key] = spreads[key][:-1]/vol

nrets = {'nominal price' : nPrice/vol, 'real price' : rPrice/vol, 'nominal total' : nTotal/vol, 'real total' : rTotal/vol}

for key in nrets:
    print('Regression for Returns vs Earnings Yield and Spreads', key, '\n\n')
    returns = nrets[key]
    Reg1 = OLS(returns, DFreg).fit()
    print(Reg1.summary())
    res = Reg1.resid
    plots(res, key + ' Model 1')
    analysis(res, key + ' Model 1')
    print('Regression for Returns without Earnings Yield', key, '\n\n')
    Reg2 = OLS(returns, DFreg[['const', 'vol', 'BAA-AAA', 'AAA-Long', 'Long-Short']]).fit()
    print(Reg2.summary())
    res = Reg2.resid
    plots(res, key + ' Model 2')
    analysis(res, key + ' Model 2')
    print('Regression for Returns without Spreads', key, '\n\n')
    Reg3 = OLS(returns, DFreg[['const', 'vol', 'EY']]).fit()
    print(Reg3.summary())
    res = Reg3.resid
    plots(res, key + ' Model 3')
    analysis(res, key + ' Model 3')
    print('Regression for Returns without Spreads and Earnings Yield', key, '\n\n')
    Reg0 = OLS(returns, DFreg[['const', 'vol']]).fit()
    print(Reg0.summary())
    res = Reg0.resid
    plots(res, key + ' Model 0')
    analysis(res, key + ' Model 0')
    
