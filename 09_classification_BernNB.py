# %% [markdown] 
## About 
#####  This code does the following:
### A. Establishes relation between X and y of permafrost data through Bernoulli Naive Bayes Classification
### B. Saves the predicted map as tiff<br/>  
### C. Prepares a side by side plot of base map and predicted permafrost map<br/>
### D. Plots an elevation band 


# %% [markdown]
## Libraries
# %% [code]
import pandas as pd, rasterio as rio, glob, numpy as np, sklearn.metrics, matplotlib.pyplot as plt, time, os, fiona, geopandas as gpd
from tqdm.notebook import tqdm as td  
from sklearn.model_selection import train_test_split
from pylab import *
import rasterio.plot
import sklearn.metrics as metrics
start = time.time()
from sklearn import preprocessing, linear_model, naive_bayes


# %% [markdown]
## Read rasters into a dataframe and preprocess
# %% [code]
def readRasters():

    # [1] Read into dataframe
    rasters = glob.glob("../07_Data/01_Raster/01_Xy_Layers/02_NoData_Set/*.tif")
    Names = ['PERMAFROST', 'DEM', 'EMISSIVITY', 'LST', 'PISR', 'WETNESS INDEX']  
    df = pd.DataFrame(columns=Names)
    for i in td(range(len(Names)), desc='Reading datasets'):
        df['{}'.format(Names[i])] = rio.open(rasters[i]).read(1).flatten()

    # [2] Add row,col info of each raster
    # row,col info is added so that pixels can be spatially located back after irregularly dropping the nans in next step
    row, col = np.zeros((1428,1445)), np.zeros((1428,1445))  # add (row, col) to XGLADS
    for a in range(1428):
        for b in range(1445):
            row[a,b], col[a,b] = a,b
    df['ROW'] = row.flatten()
    df['COL'] = col.flatten()  

    # [3] Remove nan rows
    # https://stackoverflow.com/a/36506759
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # 4 Get X and y 
    # [4.1] X
    X = df.iloc[:,1:]
    X1 = X.copy(deep=True)  # a copy of X kept
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    Names1 = ['DEM', 'EMISSIVITY', 'LST', 'PISR', 'WETNESS INDEX', 'ROW', 'COL']  
    X = pd.DataFrame(columns=Names1, data=X)
    # [4.2] y
    y = df.iloc[:,[0]]

    return X, y, rasters, X1 
X, y, rasters, X1 = readRasters()


# %% [markdown]
## [A,B] Learn relation and get out map as tiff
# %% [code]
def outMap(makeTif):

    # [1] Split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)

    # [2] train 
    regressor = naive_bayes.BernoulliNB()
    regressor.fit(XTrain, yTrain)

    # [3] predict 
    yPred = regressor.predict(X)

    # [5] yPred to spatial form 
    # [5.1] yPred to dataframe with corr. row, col info
    yPredDF = pd.DataFrame(columns=['PERMAFROST'], data=yPred, index=y.index)
    yPredDF1 = pd.concat([yPredDF, X1], axis=1)
    yPredDF1 = yPredDF1.astype({'ROW':int, 'COL':int})
    # [5.2] dataframe to spatial form
    yPredMap = np.zeros((1428, 1445))
    yPredMap[:] = np.nan
    yPredMap[yPredDF1.ROW.values, yPredDF1.COL.values] = yPredDF1.PERMAFROST.values

    # [6] save to tiff
    if makeTif==1:
        src = rio.open(rasters[0]) 
        with rio.Env():
            profile = src.profile
            with rasterio.open("../07_Data/01_Raster/02_Out_Maps/08_BernNB.tif", 'w', **profile) as dst:
                dst.write(yPredMap,1)
    print('Map saved at:', os.path.realpath("../07_Data/01_Raster/02_Out_Maps/08_BernNB.tif"))

    return yPredDF1, yPredMap 
yPredDF1, yPredMap = outMap(makeTif=1)


# %% [markdown]
## [C] Plot the base and predicted maps
# %% [code]
def plot1():

    # https://stackoverflow.com/a/13784887
    plt.rcParams["font.family"] = "Century Gothic"  # font 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.dpi = 300
    plt.subplots_adjust(right=1.2)  # shifting second subplot to right

    # [6.1] common image settings
    cmap = cm.get_cmap('binary', 3) # 3 discrete colors
    im1 = ax1.imshow(rio.open(rasters[0]).read(1), cmap=cmap)
    im2 = ax2.imshow(yPredMap, cmap=cmap)
    ax1.set_title('Reference Map', fontsize = 14)
    ax2.set_title('Predicted Map [BernNB]', fontsize = 14)
    # ax2.set_yticklabels([])
    # ax2.set_yticks([])

    # [6.1] adding common colorbar       
    cax = fig.add_axes([1.25, 0.15, 0.025, 0.7])
    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_ticks([0,1,2])  # ticks set
    cbar.set_ticklabels(['0[NP]','1[PP]','2[P]'])  # tick labels set

    # [6.3] save fig
    plt.savefig("../05_Images/03_PredMaps/08_BernNB.png", bbox_inches='tight', facecolor='w')
    plt.close()
    print('Plot saved at:', os.path.realpath("../05_Images/03_PredMaps/08_BernNB.png"))
plot1()


# %% [markdown]
## [D.1] Prepare elevation band wise data
# %% [code]
def elevationBands():
        
    # [1] set dataframe 
    # [1.1] classify permafrost values based on threshold
    df = yPredDF1
    df['PERMAFROST'] = df['PERMAFROST'].astype('float32')
    df.loc[(df["PERMAFROST"] >= -0.5) & (df["PERMAFROST"] < 0.5), "PERMAFROST"] = 0
    df.loc[(df["PERMAFROST"] >= 0.5) & (df["PERMAFROST"] < 1.5), "PERMAFROST"] = 1
    df.loc[(df["PERMAFROST"] >= 1.5) & (df["PERMAFROST"] < 2.5), "PERMAFROST"] = 2
    # [1.2] remove unclassified values
    df.drop(df[(df.PERMAFROST !=0) & (df.PERMAFROST !=1) & (df.PERMAFROST!=2)].index, inplace = True)

    # [2] Add elevation bins to each row
    bins = [4500,4750,5000,5250,5500,5750,6000,6250,6500]
    df['ElevationBins'] = (pd.cut(df['DEM'], bins))

    # [3] Split permafrost col into new cols 
    temp = pd.get_dummies(data=df.PERMAFROST)  
    df = pd.concat([df, temp], axis=1)  
    df.rename(columns={0:'NoPerm',1:'Probable',2:'Likely'}, inplace=True)  # new cols 
    df.drop(columns='PERMAFROST', inplace=True)  # original col dropped

    # [4] elev band wise LST, PISR and Permafrost categories 
    # [4.1] Mean of LST and PISR by elevation bands
    df1 = df.groupby(['ElevationBins']).mean()  
    # [4.2] Row count of three permafrost categories by elevation bands 
    df = df.groupby(['ElevationBins']).sum()
    # [4.3] Convert grid count to area (in sq. km)
    df['NoPerm'] = df['NoPerm'] * 900 / 10**6
    df['Probable'] = df['Probable'] * 900 / 10**6
    df['Likely'] = df['Likely'] * 900 / 10**6

    return df, df1    
df, df1 = elevationBands()


# %% [markdown]
## [D.2] Plot the elevation band
# %% [code]
def plot2():

    # 1 define axes
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["font.family"] = "Century Gothic"  # font 
    plt.rcParams['hatch.linewidth'] = 0.25
    fig, ax1 = plt.subplots()
    # ax1.get_yaxis().set_ticks([])

    # 2 plot permafrost categories
    width=0.25
    ax1.barh(np.arange(8), df.NoPerm, width, color='white', edgecolor='k', linewidth=0.5, label='NP')
    ax1.barh(np.arange(8)+width, df.Probable, width, color='mistyrose', edgecolor='k', hatch='//////', label='PP', linewidth=0.5)
    ax1.barh(np.arange(8)+width+width, df.Likely, width, color='aliceblue', hatch='///////', edgecolor='k', label='P', linewidth=0.5)
    ax1.set(xlabel='Area (sq. km)', ylabel='Elevation (m)', yticks=np.arange(8) + width, yticklabels=df.index, ylim=[2*width - 1, len(df)])
    ax1.set_xlim(0, 250)

    # 3 plot LST
    ax2 = ax1.twiny()
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

    # 3 plot PISR
    ax3 = ax1.twiny()
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

    # 4 combine the legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines+lines2+lines3, labels+labels2+labels3, loc=7, fontsize='small')

    # 5 save figure
    plt.savefig("../05_Images/04_Elev_Bands_Pred/08_BernNB.png", facecolor='w', bbox_inches='tight')
    plt.close()
    print('Plot saved at:', os.path.realpath("../05_Images/04_Elev_Bands_Pred/08_BernNB.png"))
    print('Time elapsed: ', np.round(time.time()-start,2), 'secs')
plot2()


# %%