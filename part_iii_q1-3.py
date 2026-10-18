import pandas as pd
import numpy as np
import os

## Question 1: Read book-to-market and stock return data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
bm = pd.read_csv('wrds_book_to_market_1970_2025.csv',
                 usecols=['permno', 'public_date', 'bm'])
cols = ['PERMNO', 'MthCalDt', 'MthRet', 'MthCap', 'PrimaryExch',
        'ConditionalType', 'TradingStatusFlg', 'USIncFlg', 'IssuerType',
        'SecurityType', 'SecuritySubType', 'ShareType']
crsp = pd.read_csv('wrds_crsp_ciz_monthly_1970_2025.csv', usecols=cols)
crsp.columns = crsp.columns.str.lower()

## Question 2: Filter stocks and merge by PERMNO and month
# Exchange codes follow the supplied WRDS instructions.
keep = (crsp.primaryexch.isin(['N', 'A', 'R'])
        & (crsp.conditionaltype == 'RW') & (crsp.tradingstatusflg == 'A')
        & (crsp.usincflg == 'Y') & crsp.issuertype.isin(['ACOR', 'CORP'])
        & (crsp.securitytype == 'EQTY') & (crsp.securitysubtype == 'COM')
        & (crsp.sharetype == 'NS'))
crsp = crsp.loc[keep].copy()
crsp['month'] = pd.to_datetime(crsp.mthcaldt).dt.to_period('M')
bm['month'] = pd.to_datetime(bm.public_date).dt.to_period('M')
bm = bm[bm.bm > 0].drop_duplicates(['permno', 'month'])
crsp = crsp.dropna(subset=['mthret', 'mthcap'])
crsp = crsp[crsp.mthcap > 0].sort_values(['permno', 'month', 'mthcap'])
crsp = crsp.drop_duplicates(['permno', 'month'], keep='last')
data = bm.merge(crsp[['permno', 'month', 'mthret', 'mthcap']],
                on=['permno', 'month'])

## Question 3: Form four portfolios at each quarter-end
rows = []
quarters = pd.period_range('1970-03', '2025-12', freq='M')[::3]
for q in quarters:
    stocks = data[data.month == q].copy()
    # Split stocks into the largest 50% and smallest 50%.
    size = stocks.mthcap.rank(pct=True, method='first')
    stocks['size'] = np.where(size > .5, 'high', 'low')

    for s in ['high', 'low']:
        group = stocks[stocks['size'] == s].copy()
        # Within each size group: top 30% BM = value, bottom 30% = growth.
        rank = group.bm.rank(pct=True, method='first')
        for style, selected in [('value', rank > .7), ('growth', rank <= .3)]:
            weights = group.loc[selected, ['permno', 'mthcap']].copy()
            # Use quarter-end value weights for the next three months.
            weights['w'] = weights.mthcap / weights.mthcap.sum()
            for k in [1, 2, 3]:
                # Holding months also require valid BM and market cap.
                hold = data.loc[data.month == q + k, ['permno', 'mthret']]
                hold = hold.merge(weights[['permno', 'w']], on='permno')
                if not hold.empty:
                    ret = (hold.w * hold.mthret).sum() / hold.w.sum()
                    rows.append([q + k, s + '_' + style, ret])

returns = pd.DataFrame(rows, columns=['month', 'portfolio', 'ret'])
returns = returns.pivot(index='month', columns='portfolio', values='ret')
returns = returns[['high_value', 'high_growth', 'low_value', 'low_growth']]
print(f'Sample: {returns.index.min()} to {returns.index.max()}, {len(returns)} months')
print('January-March 1970 are missing; earlier formation data are needed.')

## Q1: Average monthly returns of the four portfolios
print('\nQ1: Average monthly returns (%)')
print((returns.mean() * 100).round(4))

## Q2: Standard deviations of monthly portfolio returns
print('\nQ2: Monthly standard deviations (%)')
print((returns.std(ddof=1) * 100).round(4))

## Q3: Long low-value and short high-growth
returns['zero_dollar'] = returns.low_value - returns.high_growth
print('\nQ3: Zero-dollar portfolio (low-value minus high-growth)')
print(f'Average monthly return: {returns.zero_dollar.mean() * 100:.4f}%')
print(f'Monthly standard deviation: {returns.zero_dollar.std(ddof=1) * 100:.4f}%')
