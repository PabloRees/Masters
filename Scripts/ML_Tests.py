import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier,SGDRegressor, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
plt.style.use('seaborn')
from dataclasses import dataclass

@dataclass
class classificationScoreHolder:
    def __init__(self,accuracy,precision,recall,confusionMatrix,binaryConfusionMatrix ,roc_curve, best_params,roc_auc_score, binary):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.confusionMatrix = confusionMatrix
        self.binaryConfusionMatrix = binaryConfusionMatrix
        self.roc_curve = roc_curve
        self.best_params = best_params
        self.binary = binary
        self.roc_auc_score = roc_auc_score

@dataclass
class regressionScoreHolder:
    def __init__(self,MSE, MAE):
        self.MSE = MSE
        self.MAE = MAE

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

    if 'PVDM' in vecList:

        for i in range(0,20):
            DocVec20.append((f'PVDM_{i}'))

    if 'PVDBOW' in vecList:

        for i in range(0,20):
            DocVec20.append((f'PVDBOW_{i}'))


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

    Y = 'logDif_date_resid'  # options: 'DlogDif', 'logDif', 'logDif_date_resid'
    X = X_control + X_meta + X_test  # Options: any combination of X_auto, X_meta and X_NLP

    return X_control, X_meta, X_test, Y

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('Threshold')
    plt.legend()

def plot_roc_curve(Y, y_predict,Name, label=None ):

    fpr, tpr, thresholds = roc_curve(Y, y_predict)

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    plt.xlabel('False positive Rate (Sensitivity)')
    plt.ylabel('True positive Rate (Recall)')
    plt.title(f'{Name} ROC curve')

    current_roc_curve = plt

    return current_roc_curve

def Y_cat_format(df,YVar,binary:bool):
    Y_mean = np.mean(df[YVar])

    if binary:
        Y_binary = []
        for i in df[YVar]:
            if i >Y_mean:
                Y_binary.append(1)
            else:Y_binary.append(0)
        return Y_binary

    else:
        Y_sd = np.std(df[YVar])
        Y = []
        for i in df[YVar]:
            if i > Y_mean + 2 * Y_sd:
                Y.append(8)

            elif i > Y_mean + Y_sd:
                Y.append(7)

            elif i > Y_mean + 0.5 * Y_sd:
                Y.append(6)

            elif i > Y_mean:
                Y.append(5)

            elif i < (Y_mean - 2 * Y_sd):
                Y.append(1)

            elif i < (Y_mean - Y_sd):
                Y.append(2)

            elif i < (Y_mean - 0.5 * Y_sd):
                Y.append(3)

            else:
                Y.append(4)

        return Y

def dataSetup(full_df,startDate,remove_duplicate_dates=False):

    for i in full_df.columns:
        if 'Unnamed' in str(i):
            full_df.drop(str(i), axis=1, inplace=True)

    small_Df = full_df[~(full_df['Date'] < startDate)]
    small_Df.reset_index(inplace=True)
    small_Df.sort_values(by='Date')

    if remove_duplicate_dates:
        small_Df.drop_duplicates(subset='Date',inplace=True)

    trainDf, testDf = train_test_split(small_Df,train_size=0.7,random_state=42)
    testDf, valDf = train_test_split(testDf,train_size=0.5,random_state=42) #shuffling is fine because YVar is stationary

    return trainDf, testDf, valDf

def setClassifier(ML_type,clf_type,XVars,binary, random_state=42):

    clf_types = ['clf_SGD', 'clf_MLP', 'clf_NN', 'clf_KNN', 'clf_logreg', 'clf_tree', 'clf_forrest',
                 'clf_GradientBoosting']
    if not clf_type in clf_types:
        raise ValueError(f"Classifier ML_type chosen but non-classifier model chosen. Please choose from:\n{clf_types}")

    if clf_type == 'clf_SGD':
        clf = SGDClassifier(random_state=random_state, shuffle=True, loss='log', max_iter=10000)

    elif clf_type == 'clf_MLP':
        clf = MLPClassifier(max_iter=1000, shuffle=True, learning_rate_init=0.01, learning_rate='adaptive',
                            random_state=random_state)
    elif clf_type == 'clf_NN':
        inputVars = len(XVars)
        if inputVars / 5 < 8:
            L_1 = 8
        else:
            L_1 = round(inputVars / 8)

        if binary:
            L_3 = 2
        else:
            L_3 = 7

        clf = MLPClassifier(hidden_layer_sizes=(L_1, 8, L_3), activation='relu', solver='sgd',
                            max_iter=10000, random_state=random_state)

    elif clf_type == 'clf_KNN':
        clf = KNeighborsClassifier(weights='distance', algorithm='auto', n_jobs=-1)

    elif clf_type == 'clf_logreg':
        clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=10000,
                                 random_state=random_state)
    elif clf_type == 'clf_tree':
        clf = DecisionTreeClassifier(random_state=random_state)

    elif clf_type == 'clf_forrest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                     max_samples=None, max_depth=5)

    else : #clf_type == 'clf_GradientBoosting'
        clf = GradientBoostingClassifier(random_state=random_state,learning_rate=0.01,
                                        min_samples_split=0.025,min_samples_leaf=0.01,max_depth=8)

    return clf

def setRegressor(ML_type,reg_type,XVars,random_state=42):

    reg_types = ['reg_SGD', 'reg_NN', 'reg_MLR','reg_GradientBoosting']

    if not reg_type in reg_types:
        raise ValueError(f"Regressor ML_type chosen but non-regressor model chosen. Please choose from:\n{reg_types}")

    if reg_type == 'reg_GradientBoosting':
        reg = GradientBoostingRegressor(random_state=random_state, loss='squared_error',learning_rate=0.1,
                                        min_samples_split=0.05,min_samples_leaf=0.02,max_depth=3)

    elif reg_type == 'reg_MLR':
        reg = LinearRegression(fit_intercept=False,n_jobs=-1)

    elif reg_type == 'reg_NN':

        inputVars = len(XVars)
        if inputVars / 5 < 8:
            L_1 = 8
        else:
            L_1 = round(inputVars / 8)


        reg = MLPRegressor(hidden_layer_sizes=(L_1, 8, 1), activation='relu', solver='sgd',
                            max_iter=10000, random_state=random_state)

    else : #reg_type == 'reg_SGD'
        reg = SGDRegressor(random_state=random_state, shuffle=True, loss='squared_error', max_iter=10000,
                           fit_intercept=False)

    return reg

def setGridsearch(GS_params, GS_clf_type, random_state =42):

    GS_clf_types = ['GSforrest', 'GSNN', 'GStree']

    if GS_params == None:
        raise ValueError(f'CS_Gridsearch ML_type chosen. Please parse GS_params as a dictionary of matching parameters.'
                         f'\nSee sklearn documentation for parameter options')

    if GS_clf_type not in GS_clf_types:
        raise ValueError(f'CS_Gridsearch ML_type chosen. Please parse a classifier type. '
                         f'Options: {GS_clf_types}')

    if GS_clf_type == 'GS_forrest':
        GS_clf = RandomForestClassifier(random_state=random_state)

    elif GS_clf_types == 'GS_NN' or 'GS_MLP':
            GS_clf = MLPClassifier(random_state=random_state)

    elif GS_clf_type == 'GS_tree':
                GS_clf = DecisionTreeClassifier(random_state=random_state)

    return GS_clf

class CSClassifier:

    def __init__(self,full_df,startDate,YVar,XVars,clf,binary,remove_duplicate_dates=False):

        self.YVar = YVar
        self.XVars = XVars
        self.full_df = full_df
        self.startDate = startDate

        trainDf, testDf, valDf = dataSetup(full_df, startDate, remove_duplicate_dates=remove_duplicate_dates)

        Y_train = Y_cat_format(trainDf, YVar, binary)
        XY_train = trainDf[XVars]
        XY_train['Y_train'] = Y_train
        XY_train.dropna(inplace=True)
        X_train = XY_train[XVars]
        Y_train = XY_train['Y_train']

        self.clf = clf.fit(X_train, Y_train)
        self.X_train = X_train
        self.Y_train = Y_train
        self.trainDf = trainDf
        self.testDf = testDf
        self.valDf = valDf

        Y_test = Y_cat_format(testDf, YVar, binary)
        XY_test = testDf[XVars]
        XY_test['Y_test'] = Y_test
        XY_test.dropna(inplace=True)
        self.X_test = XY_test[XVars]
        self.Y_test = XY_test['Y_test']

        Y_val = Y_cat_format(valDf, YVar, binary)
        XY_val = valDf[XVars]
        XY_val['Y_val'] = Y_val
        XY_val.dropna(inplace=True)
        self.X_val = XY_val[XVars]
        self.Y_val = XY_val['Y_val']

    def getCSClassificationScores(self, X, Y, binary,Name, crossVals, scoring):

        Accuracy = cross_val_score(self.clf, X, Y, cv=crossVals, scoring=scoring)
        y_predict = cross_val_predict(self.clf, X, Y, cv=crossVals)
        ConfMat = confusion_matrix(Y, y_predict)

        if binary:
            Precision = precision_score(Y, y_predict)
            Recall = recall_score(Y, y_predict)
            precisions, recalls, thresholds = precision_recall_curve(Y, y_predict)
            plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            plt.title(f'{Name}')

            current_roc_curve = plot_roc_curve(Y, y_predict, Name)
            roc_curve = current_roc_curve

            BinaryconfMat = 'NA'

        else:
            BinaryconfMat = np.array([[np.sum(ConfMat[0:4, 0:4]), np.sum(ConfMat[0:4, 4:8])],
                                           [np.sum(ConfMat[4:8, 0:4]), np.sum(ConfMat[4:8, 4:8])]])
            Precision = BinaryconfMat[0, 0] / (BinaryconfMat[0, 0] + BinaryconfMat[1, 0])
            Recall = BinaryconfMat[0, 0] / (BinaryconfMat[0, 0] + BinaryconfMat[0, 1])
            Accuracy = (BinaryconfMat[0,0] + BinaryconfMat[1,1])/(BinaryconfMat[0,0] + BinaryconfMat[0,1] +
                                                                  BinaryconfMat[1,0] + BinaryconfMat[1,1])

            roc_curve = None

        scores = classificationScoreHolder(accuracy=Accuracy, precision=Precision,
                                            recall=Recall,
                                            confusionMatrix=ConfMat, roc_curve=roc_curve,
                                            binaryConfusionMatrix=BinaryconfMat, binary=binary,best_params=None,
                                            roc_auc_score=None)

        return scores

    def testCSClassifier(self, Name, binary,crossVals=3,
                        scoring='accuracy'):

        testScores = self.getCSClassificationScores(self.X_train,self.Y_train,binary=binary,Name=Name,
                                                  crossVals=crossVals,scoring=scoring)

        trainScores = self.getCSClassificationScores(self.X_test,self.Y_test,binary=binary,Name=Name,
                                                  crossVals=crossVals,scoring=scoring)

        valScores = self.getCSClassificationScores(self.X_val, self.Y_val, binary=binary, Name=Name,
                                                   crossVals=crossVals, scoring=scoring)

        return trainScores, testScores, valScores

class TSClassifier:

    def __init__(self, full_df, startDate, XVars, YVar, binary, remove_duplicate_dates=False, n_splits=5):

        if 'Unnamed: 0' in full_df.columns:
            full_df.drop('Unnamed: 0', axis=1, inplace=True)

        small_Df = full_df[~(full_df['Date'] < startDate)]
        small_Df.reset_index(inplace=True)

        if remove_duplicate_dates:
            small_Df.drop_duplicates(subset='Date', inplace=True)

        self.binary =binary
        self.XVars = XVars
        self.YVar = YVar
        self.small_Df = small_Df
        self.X_train = np.array(small_Df[XVars])
        Y_prepped = Y_cat_format(small_Df, YVar, binary)
        self.Y_train = np.array(Y_prepped)
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def testTSClassifier(self, clf):

        trainAccuracy = []
        trainPrecision = []
        trainRecall = []
        trainConfMat = []
        trainBinaryConfMat = []

        for train_index, test_index in self.tscv.split(self.small_Df):
            #This section clones the classifier, sets up the time series training folds and fits and tests the training folds
            clone_clf = clone(clf)
            X_train_folds = self.X_train[train_index]
            Y_train_folds = self.Y_train[train_index]

            XY_train_df = pd.DataFrame(X_train_folds, columns=self.XVars)
            XY_train_df['Y_train'] = pd.DataFrame(Y_train_folds)
            XY_train_df.dropna(inplace=True)
            X_train_folds = np.array(XY_train_df[self.XVars])
            Y_train_folds = np.array(XY_train_df['Y_train'])

            clone_clf.fit(X_train_folds, Y_train_folds)
            y_pred = clone_clf.predict(X_train_folds)

            confMat_current = confusion_matrix(Y_train_folds, y_pred)
            trainConfMat.append(confMat_current)

            if self.binary:
                #makes score calculations based on a binary prediction set
                n_correct = sum(y_pred == Y_train_folds)
                trainAccuracy.append(n_correct / len(y_pred))
                trainPrecision.append(precision_score(Y_train_folds, y_pred))
                trainRecall.append(recall_score(Y_train_folds, y_pred))
                precisions, recalls, thresholds = precision_recall_curve(Y_train_folds, y_pred)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'Train')

                train_roc_curve = plot_roc_curve(Y_train_folds, y_pred, 'Train')

            else: # binary == False
                #makes binary score calculations based on a non-binary prediction set
                binaryconfMat_current = np.array(
                    [[np.sum(confMat_current[0:4, 0:4]), np.sum(confMat_current[0:4, 4:8])],
                     [np.sum(confMat_current[4:8, 0:4]), np.sum(confMat_current[4:8, 4:8])]])

                trainBinaryConfMat.append(binaryconfMat_current)
                trainAccuracy.append((binaryconfMat_current[0, 0] + binaryconfMat_current[1,1]) /
                                     (binaryconfMat_current[0, 0] + binaryconfMat_current[0, 1] +
                                      binaryconfMat_current[1, 0] + binaryconfMat_current[1, 1]))

                trainPrecision.append(binaryconfMat_current[0, 0] / (binaryconfMat_current[0, 0] + binaryconfMat_current[1, 0]))
                trainRecall.append(binaryconfMat_current[0, 0] / (binaryconfMat_current[0, 0] + binaryconfMat_current[0, 1]))

                train_roc_curve = None

        trainingScores = classificationScoreHolder(accuracy=trainAccuracy, precision=trainPrecision, recall=trainRecall,
                                   confusionMatrix=trainConfMat,
                                   binaryConfusionMatrix=trainBinaryConfMat, binary=self.binary, best_params=None,
                                   roc_curve=train_roc_curve,roc_auc_score=None)

        testAccuracy = []
        testPrecision = []
        testRecall = []
        testConfMat = []
        testBinaryConfMat = []

        for train_index, test_index in self.tscv.split(self.small_Df):
            clone_clf = clone(clf)
            X_train_folds = self.X_train[train_index]
            Y_train_folds = self.Y_train[train_index]
            X_test_folds = self.X_train[test_index]
            Y_test_folds = self.Y_train[test_index]

            XY_train_df = pd.DataFrame(X_train_folds, columns=self.XVars)
            XY_train_df['Y_train'] = pd.DataFrame(Y_train_folds)
            XY_train_df.dropna(inplace=True)
            X_train_folds = np.array(XY_train_df[self.XVars])
            Y_train_folds = np.array(XY_train_df['Y_train'])

            XY_test_df = pd.DataFrame(X_test_folds, columns=self.XVars)
            XY_test_df['Y_train'] = pd.DataFrame(Y_test_folds)
            XY_test_df.dropna(inplace=True)
            X_test_folds = np.array(XY_test_df[self.XVars])
            Y_test_folds = np.array(XY_test_df['Y_train'])

            clone_clf.fit(X_train_folds, Y_train_folds)
            y_pred = clone_clf.predict(X_test_folds)

            confMat_current = confusion_matrix(Y_test_folds, y_pred)
            testConfMat.append(confMat_current)

            if self.binary:
                n_correct = sum(y_pred == Y_test_folds)
                testAccuracy.append(n_correct / len(y_pred))
                testPrecision.append(precision_score(Y_test_folds, y_pred))
                testRecall.append(recall_score(Y_test_folds, y_pred))
                precisions, recalls, thresholds = precision_recall_curve(Y_test_folds, y_pred)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'Test')

                test_roc_curve = plot_roc_curve(Y_test_folds, y_pred, 'Test')

            else: # binary == False
                binaryconfMat_current = np.array(
                    [[np.sum(confMat_current[0:4, 0:4]), np.sum(confMat_current[0:4, 4:8])],
                     [np.sum(confMat_current[4:8, 0:4]), np.sum(confMat_current[4:8, 4:8])]])

                testAccuracy.append((binaryconfMat_current[0, 0] + binaryconfMat_current[1,1]) /
                                     (binaryconfMat_current[0, 0] + binaryconfMat_current[0, 1] +
                                      binaryconfMat_current[1, 0] + binaryconfMat_current[1, 1]))

                testBinaryConfMat.append(binaryconfMat_current)
                testPrecision.append(binaryconfMat_current[0, 0] / (binaryconfMat_current[0, 0] + binaryconfMat_current[1, 0]))
                testRecall.append(binaryconfMat_current[0, 0] / (binaryconfMat_current[0, 0] + binaryconfMat_current[0, 1]))

                test_roc_curve = None

        if len(testBinaryConfMat) == 0: testBinaryConfMat = 'NA'


        testScores = classificationScoreHolder(accuracy=testAccuracy, precision=testPrecision, recall=testRecall,
                                   confusionMatrix=testConfMat,
                                   binaryConfusionMatrix=testBinaryConfMat, binary=self.binary, best_params=None,
                                   roc_curve=test_roc_curve,roc_auc_score=None)

        valScores = classificationScoreHolder(accuracy=None, precision=None, recall=None,
                                   confusionMatrix=None,
                                   binaryConfusionMatrix=None, binary=None, best_params=None,
                                   roc_curve=None,roc_auc_score=None)

        return trainingScores, testScores, valScores

class GridSearch:

    def __init__(self, full_df, clf, startDate, remove_duplicate_dates):

        if 'Unnamed: 0' in full_df.columns:
            full_df.drop('Unnamed: 0', axis=1, inplace=True)

        self.trainDf, self.testDf, self.valDf = dataSetup(full_df, startDate, remove_duplicate_dates=remove_duplicate_dates)
        self.clf = clf
        self.startDate = startDate
        self.full_df=full_df

    def runGrid_search(self, XVars, YVar, param_grid, split_type, remove_duplicate_dates,binary, n_splits=5,
                        n_jobs=-1):

        if split_type == 'CS':
            Y_train = Y_cat_format(self.trainDf, YVar, binary)
            XY_train = self.trainDf[XVars]
            XY_train['Y_train'] = Y_train
            XY_train.dropna(inplace=True)
            X_train = XY_train[XVars]
            Y_train = XY_train['Y_train']

            Y_test = Y_cat_format(self.testDf, YVar, binary)
            XY_test = self.testDf[XVars]
            XY_test['Y_test'] = Y_test
            XY_test.dropna(inplace=True)
            X_test = XY_test[XVars]
            Y_test = XY_test['Y_test']

            grid_cv = GridSearchCV(self.clf, param_grid, scoring="roc_auc", n_jobs=n_jobs, cv=n_splits)

        else:

            small_Df = self.full_df[~(self.full_df['Date'] < self.startDate)]
            small_Df.dropna(inplace=True)
            small_Df.reset_index(inplace=True)

            small_Df_train = small_Df.head(round(len(small_Df) * ((n_splits - 1) / n_splits)))
            X_train = np.array(small_Df_train[XVars])
            Y_prepped_train = Y_cat_format(small_Df_train, YVar, binary)
            Y_train = np.array(Y_prepped_train)

            small_Df_test = small_Df.tail(round(len(small_Df) / n_splits))
            X_test = np.array(small_Df_test[XVars])
            Y_prepped_test = Y_cat_format(small_Df_test, YVar, binary)
            Y_test = np.array(Y_prepped_test)

            tscv = TimeSeriesSplit(n_splits=n_splits)
            grid_cv = GridSearchCV(self.clf, param_grid, scoring="roc_auc", n_jobs=n_jobs, cv=tscv, error_score='raise')

        grid_cv.fit(X_train, Y_train)

        bestParams = grid_cv.best_params_
        bestScore = grid_cv.best_score_
        train_roc_auc_score = roc_auc_score(Y_train, grid_cv.predict(X_train))

        train_scores = classificationScoreHolder(best_params=bestParams, accuracy=bestScore, roc_auc_score=train_roc_auc_score)

        return train_scores

class CSRegressor:

    def __init__(self,full_df, startDate, YVar, XVars, reg, remove_duplicate_dates=False):

        self.YVar = YVar
        self.XVars = XVars
        self.full_df = full_df
        self.startDate = startDate

        trainDf, testDf, valDf = dataSetup(full_df, startDate, remove_duplicate_dates=remove_duplicate_dates)

        Y_train = trainDf[YVar]
        XY_train = trainDf[XVars]
        XY_train['Y_train'] = Y_train
        XY_train.dropna(inplace=True)
        X_train = XY_train[XVars]
        Y_train = XY_train['Y_train']

        self.reg = reg.fit(X_train, Y_train)
        self.X_train = X_train
        self.Y_train = Y_train
        self.trainDf = trainDf
        self.testDf = testDf
        self.valDf = valDf

        Y_test = testDf[YVar]
        XY_test = testDf[XVars]
        XY_test['Y_test'] = Y_test
        XY_test.dropna(inplace=True)
        self.X_test = XY_test[XVars]
        self.Y_test = XY_test['Y_test']

        Y_val = valDf[YVar]
        XY_val = valDf[XVars]
        XY_val['Y_val'] = Y_val
        XY_val.dropna(inplace=True)
        self.X_val = XY_val[XVars]
        self.Y_val = XY_val['Y_val']

    def getCSRegressorScores(self, X, Y,crossVals):

        Y_pred = cross_val_predict(self.reg, X, Y, cv=crossVals)
        MSE = mean_squared_error(Y,Y_pred)
        MAE =mean_absolute_error(Y,Y_pred)

        scores = regressionScoreHolder(MSE=MSE, MAE=MAE)

        return scores

    def testCSRegressor(self, crossVals=3 ):

        trainScores = self.getCSRegressorScores(self.X_train,self.Y_train,crossVals=crossVals)
        testScores = self.getCSRegressorScores(self.X_test,self.Y_test,crossVals=crossVals)
        valScores = self.getCSRegressorScores(self.X_val,self.Y_val,crossVals=crossVals)

        return trainScores, testScores, valScores

class TSRegressor:

    def __init__(self, full_df, startDate, XVars, YVar, remove_duplicate_dates=False, n_splits=5):

        if 'Unnamed: 0' in full_df.columns:
            full_df.drop('Unnamed: 0', axis=1, inplace=True)

        small_Df = full_df[~(full_df['Date'] < startDate)]
        small_Df.reset_index(inplace=True)

        if remove_duplicate_dates:
            small_Df.drop_duplicates(subset='Date', inplace=True)

        self.XVars = XVars
        self.YVar = YVar
        self.small_Df = small_Df
        self.X_train = np.array(small_Df[XVars])
        self.Y_train = np.array(small_Df[YVar])
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def testTSRegressor(self,reg):

        trainMSE = []
        trainMAE = []
        testMSE = []
        testMAE = []

        for train_index, test_index in self.tscv.split(self.small_Df):

            clone_reg = clone(reg)

            X_train_folds = self.X_train[train_index]
            Y_train_folds = self.Y_train[train_index]

            XY_train_df = pd.DataFrame(X_train_folds, columns=self.XVars)
            XY_train_df['Y_train'] = pd.DataFrame(Y_train_folds)
            XY_train_df.dropna(inplace=True)
            X_train_folds = np.array(XY_train_df[self.XVars])
            Y_train_folds = np.array(XY_train_df['Y_train'])

            X_test_folds = self.X_train[test_index]
            Y_test_folds = self.Y_train[test_index]

            XY_test_df = pd.DataFrame(X_test_folds, columns=self.XVars)
            XY_test_df['Y_test'] = pd.DataFrame(Y_test_folds)
            XY_test_df.dropna(inplace=True)
            X_test_folds = np.array(XY_test_df[self.XVars])
            Y_test_folds = np.array(XY_test_df['Y_test'])

            clone_reg.fit(X_train_folds, Y_train_folds)

            y_train_pred = clone_reg.predict(X_train_folds)
            trainMSE.append(mean_squared_error(Y_train_folds,y_train_pred))
            trainMAE.append(mean_absolute_error(Y_train_folds,y_train_pred))

            y_test_pred = clone_reg.predict(X_test_folds)
            testMSE.append(mean_squared_error(Y_test_folds,y_test_pred))
            testMAE.append(mean_absolute_error(Y_test_folds,y_test_pred))

        trainScores = regressionScoreHolder(trainMSE,trainMAE)
        testScores = regressionScoreHolder(testMSE,testMAE)
        valScores = regressionScoreHolder(None, None)

        return trainScores, testScores, valScores

def runML_tests(full_df,startDate,XVars, YVar,crossVals,scoring,ML_type,remove_duplicate_dates, binary, n_splits = 5, clf_type=None,
                reg_type=None,GS_clf_type=None ,random_state=42, GS_params=None
                ):

    if 'date' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['Date'])
        full_df.sort_values(by='date', inplace=True)
    else:
        if 'Date' in full_df.columns:
            full_df['Date'] = pd.to_datetime(full_df['Date'])
            full_df.sort_values(by='Date', inplace=True)
        else: raise ValueError(f'Dataframe must contain a "Date" or "date" column')

    if ML_type in ['CS_Classifier', 'TS_Classifier']:
        clf = setClassifier(ML_type=ML_type, clf_type=clf_type,XVars=XVars, random_state=random_state, binary=binary)

        if ML_type == 'CS_Classifier':
            classifier = CSClassifier(full_df=full_df, startDate=startDate, YVar=YVar, XVars=XVars, clf=clf,
                                      binary=binary, remove_duplicate_dates=remove_duplicate_dates)

            train_scores,test_scores,val_scores = classifier.testCSClassifier(Name='Train', crossVals=crossVals,
                                                                              scoring=scoring,binary=binary)

        else: #ML_type == 'TS_Classifier'
            classifier = TSClassifier(full_df=full_df, startDate=startDate, XVars=XVars, YVar=YVar, binary=binary,
                                      remove_duplicate_dates=remove_duplicate_dates, n_splits=n_splits)

            train_scores, test_scores, val_scores = classifier.testTSClassifier(clf=clf)


        return train_scores, test_scores, val_scores

    if ML_type in ['CS_Gridsearch', 'TS_Gridsearch']:

        GS_clf = setGridsearch(GS_params=GS_params,GS_clf_type=GS_clf_type,random_state=random_state)

        if ML_type == 'CS_Gridsearch': split_type='CS'
        else: split_type='TS'


        if not binary:
            raise ValueError("Grid search currently requires binary input")

        GS_object = GridSearch(full_df=full_df,clf=GS_clf,startDate=startDate,
                                   remove_duplicate_dates=remove_duplicate_dates)

        GS_scores = GS_object.runGrid_search(XVars=XVars,YVar=YVar,
                remove_duplicate_dates=remove_duplicate_dates, param_grid=GS_params,
                split_type=split_type, binary=binary, n_jobs=-1)

        return GS_scores

    if ML_type in ['CS_Regressor','TS_Regressor']:

        reg = setRegressor(ML_type=ML_type, reg_type=reg_type,XVars=XVars, random_state=random_state)

        if ML_type == 'CS_Regressor':
            regressor = CSRegressor(full_df=full_df, startDate=startDate, YVar=YVar, XVars=XVars, reg=reg,
                                    remove_duplicate_dates=remove_duplicate_dates)

            train_scores,test_scores,val_scores = regressor.testCSRegressor(crossVals=crossVals)

        else: #ML_type == 'TS_Regressor'
            regressor = TSRegressor(full_df=full_df, startDate=startDate, XVars=XVars, YVar=YVar,
                                      remove_duplicate_dates=remove_duplicate_dates, n_splits=n_splits)

            train_scores, test_scores, val_scores = regressor.testTSRegressor(reg=reg)

        return train_scores, test_scores, val_scores




