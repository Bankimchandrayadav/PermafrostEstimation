# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # About
# ### This notebook generates plots of permafrost, elevation and LST data. Elevation is on vertical axis
# %% [markdown]
# # [1](1) Libraries

# %%
import pandas as pd, matplotlib.pyplot as plt, numpy as np

# %% [markdown]
# # [2](2) Read data

# %%
df = pd.read_csv("../02_ExcelFiles/04_rastertopoint_wetness.csv")  # [1] read file
df.drop(columns=['pointid'], inplace=True)  # drop extra cols
dfOriginal = df  # keep a copy for later use

bins = [4500,4750,5000,5250,5500,5750,6000,6250,6500]  # [2] add a col of elevation bins: define bins
df['ElevationBins'] = (pd.cut(df['Elevation'], bins))   # add bins

dfNew = pd.get_dummies(data=df.Permafrost)  # [3] convert permafrost col from catg. to dummy form: prepare dummy var dataframe
df = pd.concat([df, dfNew], axis=1)  # combine old and new dataframes 

df.rename(columns={0:'NoPerm',1:'Probable',2:'Likely'}, inplace=True)  # [4] rename dummy cols
df.drop(columns='Permafrost', inplace=True)  # drop original col

# %% [markdown]
# # [3](3) Get mean LST and PISR in each elevation band

# %%
df1 = df.groupby(['ElevationBins']).mean()

# %% [markdown]
# # [4](4) Get grid count of each permafrost category in each elevation band

# %%
df = df.groupby(['ElevationBins']).sum()  # [1] sum

df['NoPerm'] = df['NoPerm'] * 900 / 10**6  # [2] convert grid count to area in sq. km
df['Probable'] = df['Probable'] * 900 / 10**6
df['Likely'] = df['Likely'] * 900 / 10**6

# %% [markdown]
# # [5](5) Plot
# Help from: [https://stackoverflow.com/a/50655786](https://stackoverflow.com/a/50655786)

# %%
plt.rcParams['figure.dpi'] = 300  # [1] # 1 define axes
plt.rcParams["font.family"] = "Century Gothic"  # set font
plt.rcParams['hatch.linewidth'] = 0.25
fig, ax1 = plt.subplots()

width=0.25  # [2] plot permafrost categories
ax1.barh(np.arange(8), df.NoPerm, width, color='white', edgecolor='k', linewidth=0.5, label='NP')
ax1.barh(np.arange(8)+width, df.Probable, width, color='mistyrose', edgecolor='k', hatch='//////', label='PP', linewidth=0.5)
ax1.barh(np.arange(8)+width+width, df.Likely, width, color='aliceblue', hatch='///////', edgecolor='k', label='P', linewidth=0.5)
ax1.set(xlabel='Area (sq. km)', ylabel='Elevation (m)', yticks=np.arange(8) + width, yticklabels=df.index, ylim=[2*width - 1, len(df)])
ax1.set_xlim(0, 250)

ax2 = ax1.twiny()  # [3] plot LST
ax2.spines['bottom'].set_position(("axes", -0.17))
ax2.spines['bottom'].set_color('blue')
ax2.plot(df1.LST, np.arange(8), color='blue', marker='s', label='LST', alpha=0.75)
ax2.set(xlabel='LST '+r'($^\degree$C)')
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")
ax2.xaxis.label.set_color('blue')
ax2.tick_params(axis='x', colors='blue')
ax2.set_xlim((0, 40))
ax2.set_xticks(np.arange(0,48,8))

ax3 = ax1.twiny()  # [3] plot PISR
ax3.spines['top'].set_color('red')
ax3.spines["top"].set_position(("axes", 1))
ax3.plot( df1.PISR, np.arange(8), color='red', marker='o', label='PISR')
ax3.xaxis.set_ticks_position("top")
ax3.xaxis.set_label_position("top")
ax3.set(xlabel='PISR '+ r'(W hr/m$^2$)')
ax3.xaxis.label.set_color('red')
ax3.tick_params(axis='x', colors='red')
ax3.xaxis.labelpad = 10
ax3.set_xlim(6200, 7200)

lines, labels = ax1.get_legend_handles_labels()  # [4] combine the legends
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(lines+lines2+lines3, labels+labels2+labels3, loc=7, fontsize='small')

plt.savefig("../05_Images/01_Distribution1.png", facecolor='w', bbox_inches='tight')  # [5] save figure
plt.close()


