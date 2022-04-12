import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('Threshold')
    plt.legend()

def plot_roc_curve(fpr, tpr,Name, label=None ):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.xlabel('False positive Rate (Sensitivity)')
    plt.ylabel('True positive Rate (Recall)')
    plt.title(f'{Name} ROC curve')

def Y_cat_format(df,YVar,binary = False):
    Y_mean = np.mean(df[YVar])
    Y_sd = np.std(df[YVar])

    if binary:
        Y_binary = []
        for i in df[YVar]:
            if i >Y_mean:
                Y_binary.append(1)
            else:Y_binary.append(0)
        return Y_binary

    else:
        Y = []
        for i in df[YVar]:
            if i > Y_mean + 2 * Y_sd:
                Y.append(8)
            else:
                if i > Y_mean + Y_sd:
                    Y.append(7)
                else:
                    if i > Y_mean + 0.5 * Y_sd:
                        Y.append(6)
                    else:
                        if i < -(Y_mean + 2 * Y_sd):
                            Y.append(1)
                        else:
                            if i < -(Y_mean + Y_sd):
                                Y.append(2)
                            else:
                                if i < -(Y_mean + 0.5 * Y_sd):
                                    Y.append(3)
                                else:
                                    if i > Y_mean:
                                        Y.append(5)
                                    else:
                                        Y.append(4)
        return Y

def dataSetup(full_df,startDate):
    if 'Unnamed: 0' in full_df.columns:
        full_df.drop('Unnamed: 0', axis=1, inplace=True)

    small_Df = full_df[~(full_df['Date'] < startDate)]
    small_Df.reset_index(inplace=True)

    testDf = full_df.sample(frac=0.15, random_state=25)
    df = full_df.drop(testDf.index)
    valDf = df.sample(frac=0.1, random_state=25)
    trainDf = df.drop(valDf.index)

    return trainDf, testDf, valDf

def TSdata_K_fold(Name,full_df,startDate,clf,XVars,YVar,binary = False,dataset = 'Train', n_splits=5):

    if dataset not in ['Train','Test']:
        raise ValueError("Incorrect dataset type selected. Options for TSdata_K_fold are only 'Train' and 'Test'  ")

    if 'Unnamed: 0' in full_df.columns:
        full_df.drop('Unnamed: 0', axis=1, inplace=True)

    small_Df = full_df[~(full_df['Date'] < startDate)]
    small_Df.reset_index(inplace=True)

    X_train = np.array(small_Df[XVars])

    Y_prepped = Y_cat_format(small_Df,YVar,binary)
    Y_train = np.array(Y_prepped)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    accuracy = []
    precision = []
    recall = []
    confMat = []
    binaryConfMat = []

    if dataset == 'Train':
        for train_index, test_index in tscv.split(small_Df):
            clone_clf = clone(clf)
            X_train_folds = X_train[train_index]
            Y_train_folds = Y_train[train_index]

            XY_train_df = pd.DataFrame(X_train_folds, columns=XVars)
            XY_train_df['Y_train'] = pd.DataFrame(Y_train_folds)
            XY_train_df.dropna(inplace=True)
            X_train_folds = np.array(XY_train_df[XVars])
            Y_train_folds = np.array(XY_train_df['Y_train'])

            clone_clf.fit(X_train_folds, Y_train_folds)
            y_pred = clone_clf.predict(X_train_folds)
            n_correct = sum(y_pred == Y_train_folds)
            accuracy.append(n_correct / len(y_pred))
            confMat_current = confusion_matrix(Y_train_folds, y_pred)
            confMat.append(confMat_current)

            if binary:
                precision.append(precision_score(Y_train_folds,y_pred))
                recall.append(recall_score(Y_train_folds,y_pred))
                precisions, recalls, thresholds = precision_recall_curve(Y_train_folds, y_pred)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'{Name}')
                plt.show()

                fpr, tpr, thresholds = roc_curve(Y_train_folds, y_pred)
                plot_roc_curve(fpr, tpr, Name)
                plt.show()


            else:
                binaryconfMat_current = np.array([[np.sum(confMat_current[0:4, 0:4]), np.sum(confMat_current[0:4, 4:8])],
                                        [np.sum(confMat_current[4:8, 0:4]), np.sum(confMat_current[4:8, 4:8])]])

                binaryConfMat.append(binaryconfMat_current)
                precision.append(confMat_current[0,0]/(confMat_current[0,0] + confMat_current[1,0] ))
                recall.append(confMat_current[0,0]/(confMat_current[0,0]+confMat_current[0,1]))


    if dataset == 'Test':
        accuracy = []
        for train_index, test_index in tscv.split(small_Df):
            clone_clf = clone(clf)
            X_train_folds = X_train[train_index]
            Y_train_folds = Y_train[train_index]
            X_test_folds = X_train[test_index]
            Y_test_folds = Y_train[test_index]

            XY_train_df = pd.DataFrame(X_train_folds, columns=XVars)
            XY_train_df['Y_train'] = pd.DataFrame(Y_train_folds)
            XY_train_df.dropna(inplace=True)
            X_train_folds = np.array(XY_train_df[XVars])
            Y_train_folds = np.array(XY_train_df['Y_train'])

            XY_test_df = pd.DataFrame(X_test_folds, columns=XVars)
            XY_test_df['Y_train'] = pd.DataFrame(Y_test_folds)
            XY_test_df.dropna(inplace=True)
            X_test_folds = np.array(XY_test_df[XVars])
            Y_test_folds = np.array(XY_test_df['Y_train'])

            print(f'Train length: {len(Y_train_folds)} --- Test length:{len(Y_test_folds)}')
            print(f'Train variance: {np.var(Y_train_folds)} --- Test variance:{np.var(Y_test_folds)}')

            clone_clf.fit(X_train_folds, Y_train_folds)
            y_pred = clone_clf.predict(X_test_folds)
            n_correct = sum(y_pred == Y_test_folds)
            accuracy.append(n_correct / len(y_pred))
            confMat_current = confusion_matrix(Y_test_folds, y_pred)
            confMat.append(confMat_current)


            if binary:
                precision.append(precision_score(Y_test_folds,y_pred))
                recall.append(recall_score(Y_test_folds,y_pred))
                precisions, recalls, thresholds = precision_recall_curve(Y_test_folds, y_pred)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'{Name}')
                plt.show()

                fpr, tpr, thresholds = roc_curve(Y_test_folds, y_pred)
                plot_roc_curve(fpr, tpr, Name)
                plt.show()


            else:
                binaryconfMat_current = np.array([[np.sum(confMat_current[0:4, 0:4]), np.sum(confMat_current[0:4, 4:8])],[np.sum(confMat_current[4:8, 0:4]), np.sum(confMat_current[4:8, 4:8])]])

                binaryConfMat.append(binaryconfMat_current)
                precision.append(confMat_current[0,0]/(confMat_current[0,0] + confMat_current[1,0] ))
                recall.append(confMat_current[0,0]/(confMat_current[0,0]+confMat_current[0,1]))

    if len(binaryConfMat) == 0 : binaryConfMat = 'NA'

    scores_df = pd.DataFrame({'accuracy':accuracy,'precision':precision, 'recall':recall, 'confMat':confMat, 'binaryConfMat':binaryConfMat})

    return scores_df

def runClassifier(Name,clf, full_df, startDate,XVars,YVar,crossVals=3,scoring='accuracy',binary=False, dataset = 'Train'):

    trainDf, testDf, valDf = dataSetup(full_df,startDate)

    Y_train = Y_cat_format(trainDf, YVar, binary)
    XY_train = trainDf[XVars]
    XY_train['Y_train'] = Y_train
    XY_train.dropna(inplace=True)
    X_train = XY_train[XVars]
    Y_train = XY_train['Y_train']

    clf.fit(X_train, Y_train)

    if dataset == 'Train':

        accuracy = cross_val_score(clf, X_train, Y_train, cv=crossVals, scoring=scoring)

        y_train_predict = cross_val_predict(clf, X_train, Y_train, cv=crossVals)
        confMat = confusion_matrix(Y_train, y_train_predict)

        if binary:
            precision = precision_score(Y_train,y_train_predict)
            recall = recall_score(Y_train,y_train_predict)
            precisions, recalls, thresholds = precision_recall_curve(Y_train, y_train_predict)
            plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            plt.title(f'{Name}')

            plt.show()

            fpr, tpr, thresholds = roc_curve(Y_train, y_train_predict)
            plot_roc_curve(fpr, tpr, Name)
            plt.show()

            return accuracy, precision, recall, confMat, 'NA'

        else:
            binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                      [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
            precision = confMat[0,0]/(confMat[0,0] + confMat[1,0] )
            recall = confMat[0,0]/(confMat[0,0]+confMat[0,1])

        return accuracy, precision, recall, confMat, binaryconfMat

    if dataset == 'Test':
        Y_test = Y_cat_format(testDf, YVar, binary)
        XY_test = testDf[XVars]
        XY_test['Y_test'] = Y_test
        XY_test.dropna(inplace=True)
        X_test = XY_test[XVars]
        Y_test = XY_test['Y_test']

        print(f'Train length: {len(Y_train)} --- Test length:{len(Y_test)}')
        print(f'Train variance: {np.var(Y_train)} --- Test variance:{np.var(Y_test)}')

        accuracy = cross_val_score(clf, X_test, Y_test, cv=crossVals, scoring=scoring)

        y_test_predict = cross_val_predict(clf, X_test, Y_test, cv=crossVals)
        confMat = confusion_matrix(Y_test, y_test_predict)

        if binary:
            precision = precision_score(Y_test,y_test_predict)
            recall = recall_score(Y_test,y_test_predict)
            precisions, recalls, thresholds = precision_recall_curve(Y_test, y_test_predict)
            plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            plt.title(f'{Name}')
            plt.show()

            fpr, tpr, thresholds = roc_curve(Y_test, y_test_predict)
            plot_roc_curve(fpr, tpr, Name)
            plt.show()

            return np.max(accuracy), precision, recall, confMat, 'NA'

        else:
            binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                      [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
            precision = confMat[0,0]/(confMat[0,0] + confMat[1,0] )
            recall = confMat[0,0]/(confMat[0,0]+confMat[0,1])

        return accuracy, precision, recall, confMat, binaryconfMat

    if dataset == 'Validation':
        Y_val = Y_cat_format(valDf, YVar, binary)
        XY_val = valDf[XVars]
        XY_val['Y_val'] = Y_val
        XY_val.dropna(inplace=True)
        X_val = XY_val[XVars]
        Y_val = XY_val['Y_val']

        accuracy = cross_val_score(clf, X_val, Y_val, cv=crossVals, scoring=scoring)

        y_val_predict = cross_val_predict(clf, X_val, Y_val, cv=crossVals)
        confMat = confusion_matrix(Y_val, y_val_predict)


        if binary:
            precision = precision_score(Y_val,y_val_predict)
            recall = recall_score(Y_val,y_val_predict)
            precisions, recalls, thresholds = precision_recall_curve(Y_val, y_val_predict)
            plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            plt.title(f'{Name}')
            plt.show()

            fpr, tpr, thresholds = roc_curve(Y_val, y_val_predict)
            plot_roc_curve(fpr, tpr, Name)
            plt.show()

            return np.max(accuracy), precision, recall, confMat, 'NA'

        else:
            binaryconfMat = np.array([[np.sum(confMat[0:4,0:4]),np.sum(confMat[0:4,4:8])],[np.sum(confMat[4:8,0:4]),np.sum(confMat[4:8,4:8])]])
            precision = confMat[0,0]/(confMat[0,0] + confMat[1,0] )
            recall = confMat[0,0]/(confMat[0,0]+confMat[0,1])

        return accuracy, precision, recall, confMat, binaryconfMat

def runGrid_search(clf, full_df, startDate,XVars,YVar,param_grid,split_type,n_splits=5,binary=False,n_jobs=2):
    trainDf, testDf, valDf = dataSetup(full_df,startDate)

    if split_type=='CS':
        Y_train = Y_cat_format(trainDf, YVar, binary)
        XY_train = trainDf[XVars]
        XY_train['Y_train'] = Y_train
        XY_train.dropna(inplace=True)
        X_train = XY_train[XVars]
        Y_train = XY_train['Y_train']

        Y_test = Y_cat_format(testDf, YVar, binary)
        XY_test = testDf[XVars]
        XY_test['Y_test'] = Y_test
        XY_test.dropna(inplace=True)
        X_test = XY_test[XVars]
        Y_test = XY_test['Y_test']

        print(f'Cross sectional parameter analysis:')
        print(f'Train length: {len(Y_train)} --- Test length:{len(Y_test)}')
        print(f'Train variance: {np.var(Y_train)} --- Test variance:{np.var(Y_test)}')

        grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=n_jobs, cv=n_splits).fit(X_train, Y_train)

        print("Param for GS", grid_cv.best_params_)
        print("CV score for GS", grid_cv.best_score_)
        print("Train AUC ROC Score for GS: ", roc_auc_score(Y_train, grid_cv.predict(X_train)))
        print("Test AUC ROC Score for GS: ", roc_auc_score(Y_test, grid_cv.predict(X_test)))

    if split_type =='TS':
        if 'Unnamed: 0' in full_df.columns:
            full_df.drop('Unnamed: 0', axis=1, inplace=True)

        small_Df = full_df[~(full_df['Date'] < startDate)]
        small_Df.dropna(inplace=True)
        small_Df.reset_index(inplace=True)

        small_Df_train = small_Df.head(round(len(small_Df) * (n_splits-1/n_splits)))
        X_train = np.array(small_Df_train[XVars])
        Y_prepped_train = Y_cat_format(small_Df_train, YVar, binary)
        Y_train = np.array(Y_prepped_train)

        small_Df_test = small_Df.tail(round(len(small_Df)/n_splits))
        X_test = np.array(small_Df_test[XVars])
        Y_prepped_test = Y_cat_format(small_Df_test, YVar, binary)
        Y_test = np.array(Y_prepped_test)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=n_jobs, cv=tscv, error_score='raise')

        grid_cv.fit(X_train,Y_train)

        print(f'Time series parameter analysis:')
        print("Param for GS", grid_cv.best_params_)
        print("CV score for GS", grid_cv.best_score_)
        print("Train AUC ROC Score for GS: ", roc_auc_score(Y_train, grid_cv.predict(X_train)))
        print("Test AUC ROC Score for GS: ", roc_auc_score(Y_test, grid_cv.predict(X_test)))

full_df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(106774, 237).csv')
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values(by='Date', inplace=True)

X_auto = ['DlogDif_1', 'DlogDif_2', 'absDlogDif_1', 'blackSwan_SD3_1', 'blackSwan_SD4_1', 'blackSwan_SD5_1',
              'stdVol_1DateResid', 'pos_neg_transform']

vars = []
for i in range(0, 199):
    vars.append(f'V_{i}')
vader = ['VaderNeg', 'VaderNeu', 'VaderPos', 'VaderComp']
blob = ['blobPol', 'blobSubj']

X_NLP = vars + vader + blob #all sets seem to have predictive value

meta_dr_1 = ['Nasdaq_dr_1','Oil_dr_1','SSE_dr_1',
              'USDX_dr_1','VIX_dr_1'] #'BTC_dr_1',

meta_ld_1= ['Nasdaq_ld_1','Oil_ld_1', 'SSE_ld_1',
              'USDX_ld_1', 'VIX_ld_1']#'BTC_ld_1', #BTC seems to be a strong predictor although it may just be shrinking the dataset and causing over fitting

X_meta = meta_ld_1 #lr performs better for the validation and train sets, they perform the same for the test set

Y = ['logDif_date_resid'] #options: 'DlogDif', 'logDif', 'logDif_date_resid'
X = X_auto + X_meta + X_NLP #Options: any combination of X_auto, X_meta and X_NLP
binary = True

inputVars = len(X)
if inputVars/5 < 8:
    L_1 = 8
else:
    L_1 = round(inputVars/8)

if binary:
    L_3 = 1
else: L_3 = 7

sgd_clf = SGDClassifier(random_state=42, shuffle=False,loss='log',max_iter=10000)
mlp_clf = MLPClassifier(max_iter=1000, shuffle=False,learning_rate_init=0.01, learning_rate='adaptive',random_state=42)
NN_clf = MLPClassifier(hidden_layer_sizes= (L_1,8,L_3), activation='relu', solver='adam', max_iter=1000, random_state=42)
knn_clf = KNeighborsClassifier(weights='distance', algorithm='auto',n_jobs=-1)
logreg_clf = LogisticRegression(solver = 'lbfgs',penalty='l2',max_iter=10000,random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)
forrest_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_samples=None, max_depth=5)
#stack_clf = StackingClassifier(random_state=42)


startDate = '2010-01-01'
crossVals = 5
scoring = 'accuracy' #'accuracy'
dataset = 'Test' #Options: Train, Test, Validation
clf = forrest_clf #options: sgd_clf, mlp_clf,NN_clf
reg = ''
MLType = 'Grid_Search' #Options: 'CS_Classifier', 'Regression', 'TS_Classifier', 'Grid_Search'

GSForrestclf = RandomForestClassifier(random_state=42)
GSNNclf = MLPClassifier(random_state=42, activation='relu', solver='adam', max_iter=1000)

#parameters for performing grid search. Parameters must match GSclf or GSreg type.
tree_param = {
    "max_depth": [2, 3, 5, 8],
    "min_samples_split": [2,10,20,50],
    "min_samples_leaf": [1,2,5,10]
}

NN_param = {
    "hidden_layer_sizes": [(8,7,6),(600,600),(4,4,4,4,4,4)],
    "max_iter": [10000,9000,11000]
}


Y_auto_scores = []
Y_NLP_scores = []
Y_meta_scores = []
Y_autoNLP_scores = []
Y_autometa_scores = []
Y_all_scores = []

if MLType == 'CS_Classifier':
    for i in Y:
        Y_auto_scores = runClassifier('Auto', clf, full_df, startDate, X_auto, i, crossVals=crossVals, scoring=scoring,
                             binary=binary, dataset=dataset)
        print(f'____________Auto_____________:\n Accuracy: {Y_auto_scores[0]}\n Precision: {Y_auto_scores[1]}\n Recall: {Y_auto_scores[2]}\n Confusion Matrix:\n{Y_auto_scores[3]}\n Binary Confusion Matrix:\n{Y_auto_scores[4]}\n\n')

        Y_NLP_scores = runClassifier('NLP', clf, full_df, startDate, X_NLP, i, crossVals=crossVals, scoring=scoring,
                            binary=binary, dataset=dataset)
        print(f'____________NLP_____________:\n Accuracy: {Y_NLP_scores[0]}\n Precision: {Y_NLP_scores[1]}\n Recall: {Y_NLP_scores[2]}\n Confusion Matrix:\n{Y_NLP_scores[3]}\nBinary Confusion Matrix:\n{Y_NLP_scores[4]}\n\n')

        Y_meta_scores = runClassifier('Meta', clf, full_df, startDate, X_meta, i, crossVals=crossVals, scoring=scoring,
                             binary=binary, dataset=dataset)
        print(f'____________Meta_____________:\n Accuracy: {Y_meta_scores[0]}\n Precision: {Y_meta_scores[1]}\n Recall: {Y_meta_scores[2]}\n Confusion Matrix:\n{Y_meta_scores[3]}\nBinary Confusion Matrix:\n{Y_meta_scores[4]}\n\n')

        Y_autoNLP_scores = runClassifier('Auto-NLP', clf, full_df, startDate, X_auto + X_NLP, i, crossVals=crossVals,
                                scoring=scoring, binary=binary, dataset=dataset)
        print(f'____________AutoNLP_____________:\n Accuracy: {Y_autoNLP_scores[0]}\n Precision: {Y_autoNLP_scores[1]}\n Recall: {Y_autoNLP_scores[2]}\n Confusion Matrix:\n{Y_autoNLP_scores[3]}\nBinary Confusion Matrix:\n{Y_autoNLP_scores[4]}\n\n')

        Y_autometa_scores = runClassifier('Auto-Meta', clf, full_df, startDate, X_auto + X_meta, i, crossVals=crossVals,
                                 scoring=scoring, binary=binary, dataset=dataset)
        print(f'____________AutoMeta_____________:\n Accuracy: {Y_autometa_scores[0]}\n Precision: {Y_autometa_scores[1]}\n Recall: {Y_autometa_scores[2]}\n Confusion Matrix:\n{Y_autometa_scores[3]}\nBinary Confusion Matrix:\n{Y_autometa_scores[4]}\n\n')

        Y_all_scores = runClassifier('All', clf, full_df, startDate, X_auto + X_NLP + X_meta, i, crossVals=crossVals,
                            scoring=scoring, binary=binary, dataset=dataset)
        print(          f'____________All_____________:\n Accuracy: {Y_all_scores[0]}\n Precision: {Y_all_scores[1]}\n Recall: {Y_all_scores[2]}\n Confusion Matrix:\n{Y_all_scores[3]}\nBinary Confusion Matrix:\n{Y_all_scores[4]}\n\n')

if MLType == 'TS_Classifier':
    for i in Y:
        scores = TSdata_K_fold('AutoMeta', full_df, startDate, clf, X, i,binary=binary,dataset=dataset, n_splits = crossVals)

        for i in scores.columns:
            print(f'{i}:')
            for j in range(len(scores)):
                print(f'{scores[i][j]}\n')
            print('\n')

if MLType == 'CS_Regressor':
    print()

if MLType == 'TS_Regressor':
    print()

if MLType == 'Grid_Search':

    if not binary == True:
        raise ValueError("Grid search currently requires binary input")

    for i in Y:
        runGrid_search(clf=GSForrestclf, full_df=full_df, startDate=startDate, XVars=X, YVar=i, param_grid=tree_param,split_type='TS', binary=binary, n_jobs=-1)





