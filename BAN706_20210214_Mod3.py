#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path


# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


from scipy.stats import trim_mean


# In[5]:


from statsmodels import robust


# In[6]:


import wquantiles


# In[7]:


conda install -c siboles wquantiles


# In[8]:


import wquantiles


# In[9]:


import seaborn as sns


# In[10]:


import matplotlib.pylab as plt


# In[18]:


DATA = Path('.').resolve().parents[1] / 'data'


# In[12]:


AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'KC_TAX_CSV = DATA / 'kc_tax.csv.gz'LC_LOANS_CSV = DATA / 'lc_loans.csv'AIRPORT_DELAYS_CSV = DATA / 'dfw_airline.csv'SP500_DATA_CSV = DATA / 'sp500_data.csv.gz'SP500_SECTORS_CSV = DATA / 'sp500_sectors.csv'STATE_CSV = DATA / 'state.csv'


# In[13]:


DATA = Path('.').resolve().parents[1] / 'data'AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'KC_TAX_CSV = DATA / 'kc_tax.csv.gz'LC_LOANS_CSV = DATA / 'lc_loans.csv'AIRPORT_DELAYS_CSV = DATA / 'dfw_airline.csv'SP500_DATA_CSV = DATA / 'sp500_data.csv.gz'SP500_SECTORS_CSV = DATA / 'sp500_sectors.csv'STATE_CSV = DATA / 'state.csv'


# In[16]:


DATA = Path('.').resolve() / 'data'AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'KC_TAX_CSV = DATA / 'kc_tax.csv.gz'LC_LOANS_CSV = DATA / 'lc_loans.csv'AIRPORT_DELAYS_CSV = DATA / 'dfw_airline.csv'SP500_DATA_CSV = DATA / 'sp500_data.csv.gz'SP500_SECTORS_CSV = DATA / 'sp500_sectors.csv'STATE_CSV = DATA / 'state.csv'


# In[19]:


DATA = Path('.').resolve().parents[1] / 'data'AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'


# In[20]:


AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'


# In[21]:


KC_TAX_CSV = DATA / 'kc_tax.csv.gz'


# In[22]:


LC_LOANS_CSV = DATA / 'lc_loans.csv'


# In[23]:


AIRPORT_DELAYS_CSV = DATA / 'dfw_airline.csv'


# In[24]:


SP500_DATA_CSV = DATA / 'sp500_data.csv.gz'


# In[25]:


SP500_SECTORS_CSV = DATA / 'sp500_sectors.csv'


# In[26]:


STATE_CSV = DATA / 'state.csv'


# In[29]:


state = pd.read_csv(STATE_CSV)


# In[30]:


state = pd.read_csv(STATE_CSV)


# In[34]:


DATA = Path('.').resolve() / 'data'


# In[36]:


STATE_CSV = DATA / 'state.csv'


# In[41]:


state = pd.read_csv(STATE_CSV)


# In[42]:


print(state.head(8))


# In[44]:


state = pd.read_csv(STATE_CSV)


# In[45]:


print(state['Population'].mean())


# In[46]:


print(trim_mean(state['Population'], 0.1))


# In[47]:


print(state['Population'].median())


# In[48]:


print(state['Murder.Rate'].mean())


# In[49]:


print(np.average(state['Murder.Rate'], weights=state['Population']))


# In[50]:


print(wquantiles.median(state['Murder.Rate'], weights=state['Population']))


# In[51]:


print(state['Population'].std())


# In[52]:


print(state['Population'].quantile(0.75) - state['Population'].quantile(0.25))


# In[53]:


print(robust.scale.mad(state['Population']))print(abs(state['Population'] - state['Population'].median()).median() / 0.6744897501960817)


# In[54]:


print(robust.scale.mad(state['Population']))


# In[56]:


print(abs(state['Population'] - state['Population'].median()).median() / 0.6744897501960817)


# In[57]:


print(state['Murder.Rate'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]))


# In[58]:


percentages = [0.05, 0.25, 0.5, 0.75, 0.95]


# In[59]:


df = pd.DataFrame(state['Murder.Rate'].quantile(percentages))


# In[60]:


df.index = [f'{p * 100}%' for p in percentages]


# In[62]:


print(df.transpose())


# In[63]:


ax = (state['Population']/1_000_000).plot.box(figsize=(3, 4))


# In[64]:


ax.set_ylabel('Population (millions)')


# In[65]:


plt.tight_layout()


# In[69]:


plt.show()


# In[70]:


ax = (state['Population']/1_000_000).plot.box(figsize=(3, 4))


# In[71]:


binnedPopulation = pd.cut(state['Population'], 10)


# In[72]:


print(binnedPopulation.value_counts())


# In[73]:


binnedPopulation.name = 'binnedPopulation'


# In[74]:


df = pd.concat([state, binnedPopulation], axis=1)


# In[75]:


df = df.sort_values(by='Population')


# In[76]:


groups = []


# In[77]:


for group, subset in df.groupby(by='binnedPopulation'):    groups.append({        'BinRange': group,        'Count': len(subset),        'States': ','.join(subset.Abbreviation)    })


# In[78]:


print(pd.DataFrame(groups))


# In[84]:


ax = (state['Population'] / 1_000_000).plot.hist(figsize=(4, 4))


# In[80]:


ax.set_xlabel('Population (millions)')


# In[81]:


plt.tight_layout()


# In[83]:


plt.show()


# In[85]:





# In[86]:


plt.tight_layout()
plt.show()


# In[87]:


ax = (state['Population'] / 1_000_000).plot.hist(figsize=(4, 4))
ax.set_xlabel('Population (millions)')
plt.tight_layout()
plt.show()


# In[88]:


ax = state['Murder.Rate'].plot.hist(density=True, xlim=[0, 12], 
                                    bins=range(1,12), figsize=(4, 4))
state['Murder.Rate'].plot.density(ax=ax)
ax.set_xlabel('Murder Rate (per 100,000)')
plt.tight_layout()
plt.show()


# In[90]:


dfw = pd.read_csv(AIRPORT_DELAYS_CSV)


# In[91]:


dfw = pd.read_csv(AIRPORT_DELAYS_CSV)


# In[92]:


DATA = Path('.').resolve() / 'data'


# In[93]:


dfw = pd.read_csv(AIRPORT_DELAYS_CSV)


# In[94]:


DATA = Path('.').resolve().parents[1] / 'data'


# In[95]:


dfw = pd.read_csv(AIRPORT_DELAYS_CSV)


# In[96]:


DATA = Path('.').resolve().parents[1] / 'data'


# In[97]:


AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'


# In[98]:


DFW_AIRLINE_CSV = DATA / 'dfw'


# In[100]:


dfw = pd.read_csv(AIRPORT_DELAYS_CSV)
print(100 * dfw / dfw.values.sum())


# In[101]:


dfw = pd.read_csv(AIRPORT_DELAYS_CSV)
print(100 * dfw / dfw.values.sum())


# In[106]:


ax = dfw.transpose().plot.bar(figsize=(4, 4), legend=False)
ax.set_xlabel('Cause Of Delay')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()


# In[107]:


ax = dfw.transpose().plot.bar(figsize=(4, 4), legend=False)
ax.set_xlabel('Cause Of Delay')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()


# In[108]:


sp500_sym = pd.read_csv(SP500_SECTORS_CSV)
sp500_px = pd.read_csv(SP500_DATA_CSV, index_col=0)


# In[109]:


telecomSymbols = sp500_sym[sp500_sym['sector'] == 'telecommunications_services']['symbol']


# In[110]:


telecom = sp500_px.loc[sp500_px.index >= '2012-07-01', telecomSymbols]
telecom.corr()
print(telecom)


# In[111]:


etfs = sp500_px.loc[sp500_px.index > '2012-07-01',                     sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]


# In[112]:


print(etfs.head())


# In[113]:


fig, ax = plt.subplots(figsize=(5, 4))
ax = sns.heatmap(etfs.corr(), vmin=-1, vmax=1,         
                 cmap=sns.diverging_palette(20, 220, as_cmap=True),    
                 ax=ax)
plt.tight_layout()
plt.show()


# In[114]:


from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize


# In[115]:


def plot_corr_ellipses(data, figsize=None, **kwargs):    ''' https://stackoverflow.com/a/34558488 '''    M = np.array(data)    if not M.ndim == 2:        raise ValueError('data must be a 2D array')    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'aspect':'equal'})    ax.set_xlim(-0.5, M.shape[1] - 0.5)    ax.set_ylim(-0.5, M.shape[0] - 0.5)    ax.invert_yaxis()    # xy locations of each ellipse center    xy = np.indices(M.shape)[::-1].reshape(2, -1).T    # set the relative sizes of the major/minor axes according to the strength of    # the positive/negative correlation    w = np.ones_like(M).ravel() + 0.01    h = 1 - np.abs(M).ravel() - 0.01    a = 45 * np.sign(M).ravel()    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,                           norm=Normalize(vmin=-1, vmax=1),                           transOffset=ax.transData, array=M.ravel(), **kwargs)    ax.add_collection(ec)    # if data is a DataFrame, use the row/column names as tick labels    if isinstance(data, pd.DataFrame):        ax.set_xticks(np.arange(M.shape[1]))        ax.set_xticklabels(data.columns, rotation=90)        ax.set_yticks(np.arange(M.shape[0]))        ax.set_yticklabels(data.index)    return ecm = plot_corr_ellipses(etfs.corr(), figsize=(5, 4), cmap='bwr_r')cb = fig.colorbar(m)cb.set_label('Correlation coefficient')plt.tight_layout()plt.show()


# In[116]:


ax = telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$')


# In[119]:


ax = telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$')
ax.set_xlabel('ATT (T)')
ax.set_ylabel('Verizon (VZ)')
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)
plt.tight_layout()
plt.show()
ax = telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$', alpha=0.5)
ax.set_xlabel('ATT (T)')
ax.set_ylabel('Verizon (VZ)')
ax.axhline(0, color='grey', lw=1)
print(ax.axvline(0, color='grey', lw=1))


# In[ ]:


ax = telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$')
ax.set_xlabel('ATT (T)')
ax.set_ylabel('Verizon (VZ)')
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)
plt.tight_layout()
plt.show()
ax = telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$', alpha=0.5)
ax.set_xlabel('ATT (T)')
ax.set_ylabel('Verizon (VZ)')
ax.axhline(0, color='grey', lw=1)
print(ax.axvline(0, color='grey', lw=1))


# In[ ]:


kc_tax = pd.read_csv(KC_TAX_CSV)
kc_tax0 = kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) & 
                            (kc_tax.SqFtTotLiving > 100) &
                               (kc_tax.SqFtTotLiving < 3500), :]
print(kc_tax0.shape)


# In[ ]:


ax = kc_tax0.plot.hexbin(x='SqFtTotLiving', y='TaxAssessedValue',


# In[ ]:


ax = kc_tax0.plot.hexbin(x='SqFtTotLiving', y='TaxAssessedValue',
 gridsize=30, sharex=False, figsize=(5, 4))
ax.set_xlabel('Finished Square Feet')
ax.set_ylabel('Tax Assessed Value')
plt.tight_layout()
plt.show()


# In[ ]:


ax = plt.subplots(figsize=(4, 4))
ax = sns.kdeplot(kc_tax0.SqFtTotLiving, kc_tax0.TaxAssessedValue, ax=ax)
ax.set_xlabel('Finished Square Feet')
ax.set_ylabel('Tax Assessed Value')
plt.tight_layout()
plt.show()


# In[ ]:


lc_loans = pd.read_csv(LC_LOANS_CSV)


# In[ ]:


crosstab = lc_loans.pivot_table(index='grade', columns='status',
                                aggfunc=lambda x: len(x), margins=True)
print(crosstab)


# In[127]:


df = crosstab.copy().loc['A':'G',:]
df.loc[:,'Charged Off':'Late'] = df.loc[:,'Charged Off':'Late'].div(df['All'], 
axis=0)
df['All'] = df['All'] / sum(df['All'])
perc_crosstab = df
print(perc_crosstab)


# In[128]:


airline_stats = pd.read_csv(AIRLINE_STATS_CSV)


# In[134]:


airline_stats.head()
ax = airline_stats.boxplot(by='airline', column='pct_carrier_delay',   
                           figsize=(5, 5))
ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.suptitle('')
plt.tight_layout()
plt.show()


# In[132]:


ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.suptitle('')
plt.tight_layout()
plt.show()


# In[ ]:




