# %% [markdown]
## About
### This code does the following:
### 1. Gets the data into X,y form 
### 2. Peforms a filter method of feature selection - ANOVA 
### 3. Peforms a filter method of feature selection - Kendall 
### 4. Peforms a filter method of feature selection - Mutual Information 
### 5. Plots the scores of above three methods 


# %% [markdown]
## Libraries 
# %% [code]
import glob, pandas as pd, numpy as np, matplotlib.pyplot as plt, rasterio as rio, time 
from tqdm.notebook import tqdm as td 
from sklearn import preprocessing, feature_selection, model_selection, tree
from scipy import stats
start = time.time()


# %% [markdown]
## [1](1) Get X,y 
# %% [code]
def getXy():

    # [1] read rasters into dataframe
    # [1.1] get raster list
    rasters = sorted(glob.glob("../07_Data/01_Raster/01_Xy_Layers/02_NoData_Set/*.tif"))  

    # [1.2] get into dataframe
    Names = ['PERM', 'DEM', 'EMISS', 'LST', 'PISR', 'WET_INDX']  # column names    
    df = pd.DataFrame(columns=Names)  
    for i in td(range(len(Names)), desc='Reading datasets'):  
        df['{}'.format(Names[i])] = rio.open(rasters[i]).read(1).flatten()

    # [1.3] add location info to dataframe
    row, col = np.zeros((1428,1445)), np.zeros((1428,1445))  
    for a in range(1428):
        for b in range(1445):
            row[a,b], col[a,b] = a,b
    df['ROW'] = row.flatten()
    df['COL'] = col.flatten()  

    # [1.2] remove nan rows
    df.dropna(how='any', inplace=True)  
    df.reset_index(drop=True, inplace=True)

    # [2] get X,y
    # [2.1] split X,y
    X = df.iloc[:,1:]  
    y = df.iloc[:,[0]]
    XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)  

    # [2.2] data scaling
    norm = preprocessing.MinMaxScaler().fit(XTrain)  # fit scaler on training data
    XTrainNorm = norm.transform(XTrain)  # transform training data
    XTestNorm = norm.transform(XTest)  # transform testing data

    return XTrainNorm, XTestNorm, yTrain, yTest 
XTrainNorm, XTestNorm, yTrain, yTest = getXy()


# %% [markdown]
## [2](2) ANOVA 
# %% [code]
def anova():
    
    Names1 = ['DEM', 'EMISS', 'LST', 'PISR', 'WI', 'Northing', 'Easting']
    F, pVal = feature_selection.f_classif(XTrainNorm, yTrain)
    df = pd.DataFrame(index=Names1, columns=['ANOVA', 'KENDALL', 'MI'])
    df.ANOVA = preprocessing.MinMaxScaler().fit_transform(np.array(F).reshape(-1,1))

    return Names1, df
Names1, df = anova()


# %% [markdown]
## [3](3) Kendall's concordance test  
# %% [code]
def kendal():

    for i in range(XTrainNorm.shape[1]):
        df.KENDALL[i], pVal1 = stats.kendalltau(XTrainNorm[:,i], yTrain)

    return df, pVal1
df, pVal1 = kendal()


# %% [markdown]
## [4](4) Mutual Information test
# %% [code]
def mutualinfo():

    MI = feature_selection.mutual_info_classif(XTrainNorm, yTrain)
    df.MI = MI
    df.to_excel("../02_ExcelFiles/08_FeatureSelection.xlsx")

    return MI, df 
MI, df = mutualinfo()


# %% [markdown] 
## [5](5) Plot the filter test scores
# %% [code]
def plotFilterScores():
    
    # [1] read the excel file 
    df1 = pd.read_excel("../02_ExcelFiles/08_FeatureSelection.xlsx")
    df1.set_index("Unnamed: 0", inplace=True)
    df1.index.names = [None]

    # [2] plot
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["font.family"] = "Century Gothic"  # set font
    df1.plot(kind='bar', hatch='/////', edgecolor='k', linewidth='0.5',colormap='Pastel1', xlabel='Predictors', ylabel='Filter Scores')
    # plt.savefig("../05_Images/07_FeatureSelection/01_FilterMethods.png", facecolor='w', bbox_inches='tight')
    # plt.close()

    return None 
plotFilterScores()


# %% [markdown]
## References 
#### [2.3](2.3) Mutual Information test : https://machinelearningmastery.com/feature-selection-with-numerical-input-data/