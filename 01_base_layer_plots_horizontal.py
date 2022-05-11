# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # About
# ### This notebook generates plots of permafrost, elevation and LST data. Elevation is on horizontal axis
# %% [markdown]
# # [1](1) Libraries

# %%
import pandas as pd, matplotlib.pyplot as plt

# %% [markdown]
# # [2](2) Read data

# %%
df = pd.read_csv("../02_ExcelFiles/04_rastertopoint_wetness.csv")  # [1] read file
df.drop(columns=['pointid'], inplace=True)  # drop extra cols
dfOriginal = df  # keep a copy for later use

bins = [4500,4750,5000,5250,5500,5750,6000,6250,6500]  # [2] add a col of elevation bins: define bins
df['ElevationBins'] = (pd.cut(df['Elevation'], bins))  # add bins

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

df['NoPerm'] = df['NoPerm'] * 30 / 10**6  # [2] convert grid count to area in sq. km
df['Probable'] = df['Probable'] * 30 / 10**6
df['Likely'] = df['Likely'] * 30 / 10**6

# %% [markdown]
# # [5](5) Plot
# Help from: [https://stackoverflow.com/a/50655786](https://stackoverflow.com/a/50655786)

# %%
plt.rcParams['figure.dpi'] = 300  # [1] # 1 define axes
plt.rcParams["font.family"] = "Century Gothic"  # set font 
fig, ax1 = plt.subplots() 
ax1.grid(b=True, which='major', color='k', linestyle='--', alpha=0.50)  # add grids
ax1.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.50)

df.plot(ax=ax1, kind='bar', y=['NoPerm','Probable', 'Likely'], title='Permafrost, PISR and LST Distribution', xlabel='Elevation Bands', ylabel='Area (sq. km)', colormap='Pastel1', legend=0, edgecolor='k', hatch='////')  # [2] plot permafrost categories

ax2 = ax1.twinx()  # [3] plot LST
ax2.spines['right'].set_position(('axes', 1.0))
df1.plot(marker='o', ax=ax2, y=['LST'], legend=0, ylabel='LST ' +r'($^\degree$C)')
ax2.yaxis.label.set_color('blue')
ax2.spines['right'].set_color('blue')
ax2.tick_params(axis='y', colors='blue')

ax3 = ax1.twinx()  # [4] plot PISR
ax3.yaxis.set_ticks_position('left')
ax3.yaxis.set_label_position('left')
ax3.spines['left'].set_position(('axes', -0.1))
ax3.yaxis.label.set_color('red')
ax3.spines['left'].set_color('red')
ax3.tick_params(axis='y', colors='red')
df1.plot(marker='x', ax=ax3, y=['PISR'], ylabel='PISR', legend=0, color='red')

lines, labels = ax1.get_legend_handles_labels()  # [4] combine the legends
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax3.legend(lines+lines2+lines3, labels+labels2+labels3, loc=7, fontsize='small')

plt.savefig("../05_Images/01_Distribution3.png", facecolor='w', bbox_inches='tight')  # [5] save figure
plt.close()


