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
EY = earnings/price
nPrice = np.diff(np.log(price))
rPrice = nPrice - inflation
nTotal = np.array([np.log(price[k+1] + dividend[k+1]) - np.log(price[k]) for k in range(N)])
rTotal = nTotal - inflation

rates = ['BAA', 'AAA', 'Long', 'Short']
data4 = df[rates]
for key in rates:
    plt.plot(range(1927, 2025), df[key], label = key)
    plt.xlabel('Years')
    plt.ylabel('Rates')
    plt.title('Rate Plot')
    plt.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left')
plt.savefig('allrates.png')
plt.close()

spreads = {}
spreads['BAA-AAA'] = (df['BAA'] - df['AAA']).values
spreads['AAA-Long'] = (df['AAA'] - df['Long']).values
spreads['Long-Short'] = (df['Long'] - df['Short']).values

print('Spreads VAR')
for key in spreads:
    plt.plot(range(1927, 2025), spreads[key], label = key)
plt.xlabel('Years')
plt.ylabel('Spreads')
plt.title('Spread Plot')
plt.legend(bbox_to_anchor=(0.45, 0.95), loc='upper left')
plt.savefig('spreads.png')
plt.close()

DFreg = pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'EY' : EY[:-1]})
for key in spreads:
    DFreg[key] = spreads[key][:-1]/vol

nrets = {'nominal price' : nPrice/vol, 'real price' : rPrice/vol, 'nominal total' : nTotal/vol, 'real total' : rTotal/vol}

for key in nrets:
    print('Regression for Returns', key, '\n\n')
    returns = nrets[key]
    Regression = OLS(returns, DFreg).fit()
    print(Regression.summary())
    res = Regression.resid
    plots(res, key + ' Returns Full')
    analysis(res, key + ' Returns Full')
    print('Cut Regression for Returns', key, '\n\n')
    CutRegression = OLS(returns, DFreg[['const', 'vol', 'EY']]).fit()
    print(CutRegression.summary())
    res = CutRegression.resid
    plots(res, key + ' Returns Cut')
    analysis(res, key + ' Returns Cut')