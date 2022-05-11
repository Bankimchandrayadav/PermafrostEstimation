# %% [markdown]
## About
#### This code does the following:
#### 1. Gets the data into X,y form 
#### 2. Peforms a wrapper method of feature selection - Enumerate all feature subsets 


# %% [markdown]
## Libraries 
# %% [code]
import glob, pandas as pd, numpy as np, matplotlib.pyplot as plt, rasterio as rio, time, seaborn as sns, os
from tqdm.notebook import tqdm as td 
from sklearn import preprocessing, feature_selection, model_selection, tree, discriminant_analysis, linear_model, neighbors, naive_bayes, ensemble
from matplotlib import cm
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

    # [2.2] data scaling
    norm1 = preprocessing.MinMaxScaler().fit(X)  # fit scaler on all data
    XNorm = norm1.transform(X)  # transform all data

    return XTrainNorm, XTestNorm, yTrain, yTest, XNorm, y, rasters 
XTrainNorm, XTestNorm, yTrain, yTest, XNorm, y, rasters = getXy()


# %% [markdown]
## [2](2) Enumerate All Feature Subsets
# %% [code]
def enumerateSubsets():

    # feature selection by enumerating all possible subsets of features
    from itertools import product
    from numpy import mean
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    # define dataset
    n_cols = XNorm.shape[1]
    best_subset, best_score = None, 0.0
    # enumerate all combinations of input features
    for subset in product([True, False], repeat=n_cols):
        # convert into column indexes
        ix = [i for i, x in enumerate(subset) if x]
        # check for now column (all False)
        if len(ix) == 0:
            continue
        # select columns
        X_new = XNorm[:, ix]
        # define model
        model = DecisionTreeClassifier()
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=cv, n_jobs=-1)
        # summarize scores
        result = mean(scores)
        # report progress
        print('>;f(%s) = %f ' % (ix, result))
        # check if it is better than the best so far
        if best_score is None or result >= best_score:
            # better result
            best_subset, best_score = ix, result
    # report best
    print('Done!')
    print('f(%s) = %f' % (best_subset, best_score))

    return None 
# enumerateSubsets()


# %% [markdown]
## [3](3) Recursive Feature Elimination
# %% [code]
def rfe():

    estimators = [
        linear_model.RidgeClassifier(),  #1 
        linear_model.LogisticRegression(),  #2 
        linear_model.SGDClassifier(),  #3 
        linear_model.Perceptron(tol=1e-3, random_state=0),  #4
        discriminant_analysis.LinearDiscriminantAnalysis(),  #5
        ensemble.RandomForestClassifier(n_estimators=10),  #6
        ensemble.GradientBoostingClassifier(n_estimators=10),  #7
        naive_bayes.BernoulliNB(), #8
        tree.DecisionTreeClassifier(max_depth=3), #9
        ensemble.AdaBoostClassifier(n_estimators=10)]  #10
    estimatorsName = [
        'RidgeClf',  #1
        'LogisticClf',  #2 
        'SGDC',  #3 
        'Perceptron',  #4
        'LDA',  #5
        'RF',  #6
        'GDC',  #7
        'BernoulliNB',  #8 
        'DTC',  #9
        'AdaBoost']  #10

    index = ['DEM', 'EMISS', 'LST', 'PISR', 'WI', 'Northing', 'Easting']
    dfSupport = pd.DataFrame(index=index, columns=estimatorsName)
    dfRanking = pd.DataFrame(index=index, columns=estimatorsName)

    for i in range(len(estimators)):
        selector = feature_selection.RFE(estimator=estimators[i])
        selector.fit(XTrainNorm, yTrain)
        dfSupport['{}'.format(estimatorsName[i])] = selector.support_
        dfRanking['{}'.format(estimatorsName[i])] = selector.ranking_
        print('{} done'.format(estimatorsName[i]))

    dfSupport.to_excel("../02_ExcelFiles/09_RFE_Support.xlsx")
    dfRanking.to_excel("../02_ExcelFiles/10_RFE_Rankings.xlsx")
            
    return dfSupport, dfRanking
# dfSupport, dfRanking=rfe()


# %% [markdown]
## [4](4) Plot the above scores
# %% [code]
def plot1():

    df1 = pd.read_excel("../02_ExcelFiles/09_RFE_Support.xlsx")
    df1.set_index('Unnamed: 0', inplace=True)
    df1.index.names = [None]

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["font.family"] = "Century Gothic"  # set font
    # cmap = cm.get_cmap('Dark2_r', 2)
    cmap = cm.get_cmap('Pastel2_r', 2)
    ax = sns.heatmap(df1, annot=True, cmap=cmap, linewidths=0.3, robust=True, xticklabels=0)

    # colorbar settings
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0,1])  # ticks set
    # cbar.set_ticklabels(['0 [Rej]','1 [Sel]'])  # tick labels set
    cbar.set_label('Selection Prestige')

    # save 
    plt.savefig("../05_Images/07_FeatureSelection/02_WrapperMethodsA_original", facecolor='w', bbox_inches='tight')
    plt.close()

    return None
plot1()


# %% [code]
# plot rankings
def plot2():

    df1 = pd.read_excel("../02_ExcelFiles/10_RFE_Rankings.xlsx")
    df1.set_index('Unnamed: 0', inplace=True)
    df1.index.names = [None]

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["font.family"] = "Century Gothic"  # set font
    # cmap = cm.get_cmap('Set1_r', 5)
    cmap = cm.get_cmap('Pastel2', 5)
    ax = sns.heatmap(df1, annot=True, cmap=cmap, linewidths=0.3, robust=True, xticklabels=1)

    # colorbar settings
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([1,2,3,4,5])  # ticks set
    cbar.set_label('Rankings')


    # save 
    plt.savefig("../05_Images/07_FeatureSelection/02_WrapperMethodsB_original", facecolor='w', bbox_inches='tight')
    plt.close()

    return None
plot2()



# %% [markdown]
## References 
#### [2](2) Enumerate subsets: https://machinelearningmastery.com/feature-selection-with-optimization/
