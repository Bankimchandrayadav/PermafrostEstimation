# %% [markdown] 
## About 
### This code does the following:
#### 1. Tries out different classification methods for generating permafrost maps 
#### 2. Stores the performance metrics of all methods in a dataframe
#### The classifiers selected in the following scripts are selected on the merit prepared in this script


# %% [markdown]
## Libraries 
# %% [code]
import glob, pandas as pd, rasterio as rio, numpy as np, time, matplotlib.pyplot as plt
from scipy.sparse.construct import random
from tqdm.notebook import tqdm as td
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 
from sklearn import preprocessing, linear_model, discriminant_analysis, neighbors, naive_bayes, tree, ensemble, neural_network
from sklearn.experimental import enable_hist_gradient_boosting
start = time.time()




# %% [markdown]
## Get X, y
# %% [code]
def getXy():

    # [1] read into dataframe
    rasters = glob.glob("../07_Data/01_Raster/01_Xy_Layers/02_NoData_Set/*.tif")
    Names = ['PERM', 'DEM', 'EMISS', 'LST', 'PISR', 'WET_INDX']  
    df = pd.DataFrame(columns=Names)
    for i in td(range(len(Names)), desc='Reading datasets'):
        df['{}'.format(Names[i])] = rio.open(rasters[i]).read(1).flatten()

    # [2] add row,col info of each raster
    # row,col info is added so that pixels can be spatially located back after irregularly dropping the nans in next step
    row, col = np.zeros((1428,1445)), np.zeros((1428,1445))  # add (row, col) to XGLADS
    for a in range(1428):
        for b in range(1445):
            row[a,b], col[a,b] = a,b
    df['ROW'] = row.flatten()
    df['COL'] = col.flatten()  

    # [3] remove nan rows
    # https://stackoverflow.com/a/36506759
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # [4] get X and y 
    X = df.iloc[:,1:]
    y = df.iloc[:,[0]]
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)
    # [4.1] scale data
    scaler = preprocessing.StandardScaler()
    scaler.fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XTest = scaler.transform(XTest)

    return X, y, XTrain, XTest, yTrain, yTest, rasters 
X, y, XTrain, XTest, yTrain, yTest, rasters = getXy()


# %% [markdown]
## Define the accuracy metrics dataframe
# %% [code]
dfMetrics = pd.DataFrame(index=['Accuracy', 'ClassAvgAccuracy', 'Prec0', 'Prec1', 'Prec2', 'Rec1', 'Rec2', 'Rec3', 'Del_PP', 'Del_NP', 'Del_P'])


# %% [markdown]
## [1] Ridge Classifier
# %% [code]
def ridgeCLF():
    
    # [1] Classification
    clf = linear_model.RidgeClassifier()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics 
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['Ridge'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    
    return dfMetrics
dfMetrics = ridgeCLF()


# %% [markdown]
## [2] Logistic Classifier
# %% [code]
def logisticCLF():
    
    # [1] Classification
    clf = linear_model.LogisticRegression()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics 
    # [2.1] statistical scores
    m1 =np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['Logistic'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = logisticCLF()


# %% [markdown]
## [3] SGDC Classifier
# %% [code]
def sgdcCLF():
    
    # [1] Classification
    clf = linear_model.SGDClassifier()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics 
    # [2.1] statistical scores
    m1 =np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['SGDC'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = sgdcCLF()


# %% [markdown]
## [4] Perceptron Classifier
# %% [code]
def perceptronCLF():
    
    # [1] Classification
    clf = linear_model.Perceptron(tol=1e-3, random_state=0)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics 
    # [2.1] statistical scores
    m1 =np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['Perceptron'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = perceptronCLF()


# %% [markdown]
## [5] Passive Aggressive Classifier
# %% [code]
def passAggCLF():
    
    # [1] Classification
    clf = linear_model.PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics 
    # [2.1] statistical scores
    m1 =np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['PassAgg'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = passAggCLF()


# %% [markdown]
## [6] Linear Discriminant Analysis
# %% [code]
def ldaCLF():
    
    # [1] Classification    
    clf = discriminant_analysis.LinearDiscriminantAnalysis()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics 
    # [2.1] statistical scores
    m1 =np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['LDA'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = ldaCLF()


# %% [markdown]
## [7] Quadratic Discriminant Analysis
# %% [code]
def qdaCLF():
    
    # [1] Classification    
    clf = discriminant_analysis.QuadraticDiscriminantAnalysis()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['QDA'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = qdaCLF()


# %% [markdown]
## [8] KNeighbors
# %% [code]
def kneighborCLF():

    # [1] Classification
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['KNeighbor'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = kneighborCLF()


# %% [markdown]
## [9] Gaussian Naive Bayes
# %% [code]
def gaussNBCLF():

    # [1] Classification
    clf = naive_bayes.GaussianNB()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['GaussNB'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = gaussNBCLF()


# %% [markdown]
## [10] Bernoulli Naive Bayes
# %% [code]
def bernNBCLF():

    # [1] Classification
    clf = naive_bayes.BernoulliNB()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['BernNB'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = bernNBCLF()


# %% [markdown]
## [11] Decision Trees
# %% [code]
def dtcCLF():

    # [1] Classification
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['DTC'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    plt.figure(dpi=300)
    tree.plot_tree(clf)
  
    return dfMetrics
dfMetrics = dtcCLF()


# %% [markdown]
## [12] Bagging Classifier
# %% [code]
def bagKNCLF():

    # [1] Classification
    base_estimator = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf = ensemble.BaggingClassifier(base_estimator=base_estimator)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['BaggingKNeighbor'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = bagKNCLF()


# %% [markdown]
## [13] Random Forests
# %% [code]
def rfCLF():

    # [1] Classification
    clf = ensemble.RandomForestClassifier(n_estimators=10)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['RF'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = rfCLF()


# %% [markdown]
## [14] Extra Trees Classifier
# %% [code]
def extraTreesCLF():

    # [1] Classification
    clf = ensemble.ExtraTreesClassifier(n_estimators=10)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['ExtraTrees'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = extraTreesCLF()


# %% [markdown]
## [15] AdaBoost Classifier
# %% [code]
def abCLF():

    # [1] Classification
    clf = ensemble.AdaBoostClassifier(n_estimators=10)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['AdaBoost'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = abCLF()


# %% [markdown]
## [16] GradientBoosting Classifier
# %% [code]
def gbCLF():

    # [1] Classification
    clf = ensemble.GradientBoostingClassifier(n_estimators=10)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['GradBoost'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = gbCLF()


# %% [markdown]
## [17] HistogramGradientBoosting Classifier
# %% [code]
def gradBoostCLF():

    # [1] Classification
    clf = ensemble.HistGradientBoostingClassifier()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['HistGradBoost'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = gradBoostCLF()


# %% [markdown]
## [18] HistogramGradientBoosting Classifier
# %% [code]
def histgradCLF():

    # [1] Classification
    clf = ensemble.HistGradientBoostingClassifier()
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['HistGradBoost'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = histgradCLF()


# %% [markdown]
## [19] Voting Classifier
# %% [code]
def votingCLF():

    # [1] Classification
    # [1.1] make ensemble
    clf1 = ensemble.RandomForestClassifier(n_estimators=10)
    clf2 = linear_model.LogisticRegression()
    clf3 = neighbors.KNeighborsClassifier()
    clf = ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    # [1.2] fit
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['VotingCLF'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = votingCLF()


# %% [markdown]
## [20] Stacked Classifier
# %% [code]
def stackedCLF():

    # [1] Classification
    # [1.1] make ensemble
    clf1 = ensemble.RandomForestClassifier(n_estimators=10)
    clf2 = linear_model.LogisticRegression()
    clf3 = neighbors.KNeighborsClassifier()
    clf = ensemble.StackingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], final_estimator=clf2)
    # [1.2] fit
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['StackedCLF'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = stackedCLF()


# %% [markdown]
## [21] ANN Classifier
# %% [code]
def annCLF():

    # [1] Classification
    clf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    clf.fit(XTrain, yTrain)
    yPred = clf.predict(XTest)
    yPredDF = pd.DataFrame(columns=['yPred'], data=yPred, index=yTest.index)

    # [2] Metrics
    # [2.1] statistical scores
    m1 = np.round(metrics.accuracy_score(y_pred=yPredDF, y_true=yTest, normalize=True)*100,2)
    m2 = np.round(metrics.balanced_accuracy_score(y_pred=yPredDF, y_true=yTest)*100,2)
    print(metrics.classification_report(y_true=yTest, y_pred=yPredDF))
    m31, m32, m33 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    m41, m42, m43 =  np.round(metrics.precision_score(y_true=yTest, y_pred=yPredDF, average=None),2)
    # [2.2] Delta of each class 
    delta = np.round(((yPredDF.yPred.value_counts()-yTest.PERM.value_counts())/yTest.PERM.value_counts())*100,2)
    dfMetrics['ANNCLF'] = m1,m2, m31, m32, m33, m41, m42, m43, delta[0], delta[1], delta[2]

    return dfMetrics
dfMetrics = annCLF()


# %% [markdown]
## Save the output
# %% [code]
dfMetrics.T.to_excel("../02_ExcelFiles/07_Classifiers_Metrics1.xlsx")
print('Time elapsed: ', np.round((time.time()-start)/60,2), 'mins')


# %% [markdown]
