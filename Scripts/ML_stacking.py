from ML_Tests import *

def setupXYvars():
    X_control = ['DlogDif_1', 'DlogDif_2', 'absDlogDif_1', 'blackSwan_SD3_1', 'blackSwan_SD4_1', 'blackSwan_SD5_1',
              'stdVol_1DateResid', 'pos_neg_transform']

    vars = []
    for i in range(0, 199):
        vars.append(f'V_{i}')
    vader = ['VaderNeg', 'VaderNeu', 'VaderPos', 'VaderComp']
    blob = ['blobPol', 'blobSubj']

    X_test = vars + vader + blob  # all sets seem to have predictive value

    meta_dr_1 = ['Nasdaq_dr_1', 'Oil_dr_1', 'SSE_dr_1',
                 'USDX_dr_1', 'VIX_dr_1']  # 'BTC_dr_1',

    meta_ld_1 = ['Nasdaq_ld_1', 'Oil_ld_1', 'SSE_ld_1',
                 'USDX_ld_1',
                 'VIX_ld_1']  # 'BTC_ld_1', #BTC seems to be a strong predictor although it may just be shrinking the dataset and causing over fitting

    X_meta = meta_ld_1  # lr performs better for the validation and train sets, they perform the same for the test set

    Y = ['logDif_date_resid']  # options: 'DlogDif', 'logDif', 'logDif_date_resid'
    X = X_control + X_meta + X_test  # Options: any combination of X_auto, X_meta and X_NLP

    return X_control, X_meta, X_test, Y

X_control, X_meta, X_test, Y = setupXYvars()

startDate = '2010-01-01'
crossVals = 5
scoring = 'accuracy'  # 'accuracy'
dataset = 'Test'  # Options: Train, Test, Validation
clf = forrest_clf  # options: sgd_clf, mlp_clf,NN_clf
reg = ''
MLType = 'Grid_Search'  # Options: 'CS_Classifier', 'Regression', 'TS_Classifier', 'Grid_Search'
binary = True

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

GS_clf = GStree_clf  # empty classifiers for grid search. Options: GSForrest_clf, GSNN_clf, GStree_clf
