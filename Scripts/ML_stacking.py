import pandas as pd

from ML_Tests import *
from Feature_selection import Shrinkage_Methods

def setupXYvars(vecList):
    X_control = ['DlogDif_1', 'DlogDif_2', 'absDlogDif_1', 'blackSwan_SD3_1', 'blackSwan_SD4_1', 'blackSwan_SD5_1',
              'stdVol_1DateResid', 'pos_neg_transform']

    WVec = []
    DocVec200 = []
    DocVec20 = []
    vader = []
    blob = []

    if 'WV' in vecList:

        for i in range(0, 200):
            WVec.append(f'WV_{i}')

    if 'DV_200_' in vecList:
        for i in range(0, 200):
            DocVec200.append(f'DV_200_{i}')

    if 'DV_20_' in vecList:

        for i in range(0,20):
            DocVec20.append((f'DV_20_{i}'))

    if 'vader' in vecList:
        vader = ['VaderNeg', 'VaderNeu', 'VaderPos', 'VaderComp']

    if 'blob' in vecList:
        blob = ['blobPol', 'blobSubj']

    X_test = WVec +DocVec200 + DocVec20+ vader + blob  # all sets seem to have predictive value

    meta_dr_1 = ['Nasdaq_dr_1', 'Oil_dr_1', 'SSE_dr_1',
                 'USDX_dr_1', 'VIX_dr_1']  # 'BTC_dr_1',

    meta_ld_1 = ['Nasdaq_ld_1', 'Oil_ld_1', 'SSE_ld_1',
                 'USDX_ld_1',
                 'VIX_ld_1']  # 'BTC_ld_1', #BTC seems to be a strong predictor although it may just be shrinking the dataset and causing over fitting

    X_meta = meta_ld_1  # lr performs better for the validation and train sets, they perform the same for the test set

    Y = ['logDif_date_resid']  # options: 'DlogDif', 'logDif', 'logDif_date_resid'
    X = X_control + X_meta + X_test  # Options: any combination of X_auto, X_meta and X_NLP

    return X_control, X_meta, X_test, Y

filePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(73827, 458).csv'
startDate = '2010-01-01'
crossVals = 5
scoring = 'accuracy'  # 'accuracy'
dataset = 'Test'  # Options: Train, Test, Validation
clfType = 'clf_GradientBoosting'  # options: 'clf_SGD','clf_MLP','clf_NN','clf_KNN','clf_logreg','clf_tree','clf_forrest','clf_GradientBoosting'
reg = ''
MLType = 'CS_Classifier'  # Options: 'CS_Classifier', 'Regression', 'TS_Classifier', 'Grid_Search'
binary = True
remove_duplicate_dates = False

# parameters for performing grid search. Parameters must match GSclf or GSreg type.
tree_param = {
    "max_depth": [2, 3, 5, 8],
    "min_samples_split": [2, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10]
}

NN_param = {
    "hidden_layer_sizes": [(8, 7, 6)],
    "max_iter": [10000, 100000],
    "activation": ['relu'],
    "solver": ['lbfgs']

    # "activation":['relu','tanh','logistic','identity'],
    # "solver":['lbfgs', 'sgd', 'adam']
}

#GS_clf = GStree_clf  # empty classifiers for grid search. Options: GSForrest_clf, GSNN_clf, GStree_clf

df = pd.read_csv(filePath)
print(f'There are {len(df)} entries in the df and {len(df["Date"].unique())} unique dates in the df for a ratio of {len(df["Date"].unique())/len(df)}')

#test = Shrinkage_Methods(data=df,X_variables=X_test,Y_variable=Y,num_features=12)
#print('\nLasso')
#Lasso = test.run_Lasso()
#print('\nElasticNet')
#ElasticNet = test.run_ElasticNet()
#print('\nRidge')
#Ridge = test.run_Ridge()
X_control, X_meta, X_test, Y = setupXYvars(['vader']) #options: ['WV','DV_200_', 'DV_20_','vader',blob']

#for i in range(0,10):
    #X_test.pop(0)

trainDf,testDf,valDf = dataSetup(df,startDate='1990-01-01',remove_duplicate_dates=False)
possibleVars = ['DlogDif_1', 'DlogDif_2', 'pos_neg_transform','Nasdaq_ld_1', 'Oil_ld_1','VIX_ld_1','DV_20_6','DV_20_8','DV_20_13','DV_20_15']
GS = Shrinkage_Methods(data=trainDf,X_variables=possibleVars,Y_variable=Y,num_features=12)
GS.Elastic_Gridsearch(minAlpha=0,maxAlpha=10,l1_ratio=0.3,show_coefficients=True) #for lasso l1_ratio = 1, for ridge l1_ratio =0.011

exit()

del df

runML_tests(filePath=filePath,startDate=startDate,X_control=possibleVars,X_test=X_test,X_meta=X_meta,Y=Y,
            remove_duplicate_dates=remove_duplicate_dates,
            crossVals=crossVals,scoring=scoring,dataset=dataset,clf_type=clfType,ML_type=MLType,binary=binary)