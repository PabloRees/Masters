import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
plt.style.use('seaborn')
from dataclasses import dataclass

@dataclass
class scoreHolder:
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

def setClassifier(ML_type,clf_type,XVars,random_state=42,binary=False):

    clf_types = ['clf_SGD', 'clf_MLP', 'clf_NN', 'clf_KNN', 'clf_logreg', 'clf_tree', 'clf_forrest',
                 'clf_GradientBoosting']
    if not clf_type in clf_types:
        raise ValueError(f"Classifier ML_type chosen but non-classifier model chosen. Please choose from:\n{clf_types}")

    if clf_type == 'clf_SGD':
        clf = SGDClassifier(random_state=random_state, shuffle=False, loss='log', max_iter=10000)
    elif clf_type == 'clf_MLP':
        clf = MLPClassifier(max_iter=1000, shuffle=False, learning_rate_init=0.01, learning_rate='adaptive',
                            random_state=random_state)
    elif clf_type == 'clf_NN':
        inputVars = len(XVars)
        if inputVars / 5 < 8:
            L_1 = 8
        else:
            L_1 = round(inputVars / 8)

        if binary:
            L_3 = 1
        else:
            L_3 = 7

        clf = MLPClassifier(hidden_layer_sizes=(L_1, 8, L_3), activation='relu', solver='adam',
                            max_iter=1000, random_state=random_state)

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

    elif clf_type == 'clf_GradientBoosting':
        clf = GradientBoostingClassifier()

    print(f'ML_type = {ML_type}')

    return clf

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
    else:
        if GS_clf_types == 'GS_NN' or 'GS_MLP':
            GS_clf = MLPClassifier(random_state=random_state)
        else:
            if GS_clf_type == 'GS_tree':
                GS_clf = DecisionTreeClassifier(random_state=random_state)

    return GS_clf

class CSClassifier:

    def __init__(self,full_df,startDate,YVar,XVars,clf,binary=False,remove_duplicate_dates=False):

        trainDf, testDf, valDf = dataSetup(full_df, startDate, remove_duplicate_dates=remove_duplicate_dates)

        Y_train = Y_cat_format(trainDf, YVar, binary)
        XY_train = trainDf[XVars]
        XY_train['Y_train'] = Y_train
        XY_train.dropna(inplace=True)
        X_train = XY_train[XVars]
        Y_train = XY_train['Y_train']

        self.YVar = YVar
        self.XVars = XVars
        self.full_df = full_df
        self.startDate = startDate
        self.clf = clf.fit(X_train, Y_train)
        self.X_train = X_train
        self.Y_train = Y_train
        self.trainDf = trainDf
        self.testDf = testDf
        self.valDf = valDf

    def testCSClassifier(self, Name,crossVals=3,
                        scoring='accuracy', binary=False, dataset='Train'):

        if dataset == 'Train':

            accuracy = cross_val_score(self.clf, self.X_train, self.Y_train, cv=crossVals, scoring=scoring)
            y_train_predict = cross_val_predict(self.clf, self.X_train, self.Y_train, cv=crossVals)
            confMat = confusion_matrix(self.Y_train, y_train_predict)

            if binary:
                precision = precision_score(self.Y_train, y_train_predict)
                recall = recall_score(self.Y_train, y_train_predict)
                precisions, recalls, thresholds = precision_recall_curve(self.Y_train, y_train_predict)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'{Name}')

                plt.show()

                fpr, tpr, thresholds = roc_curve(self.Y_train, y_train_predict)
                plot_roc_curve(fpr, tpr, Name)
                plt.show()

                scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall,
                                           confusionMatrix=confMat, binaryConfusionMatrix='NA', binary=binary)


            else:
                binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                          [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
                precision = confMat[0, 0] / (confMat[0, 0] + confMat[1, 0])
                recall = confMat[0, 0] / (confMat[0, 0] + confMat[0, 1])

                scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall,
                                           confusionMatrix=confMat,
                                           binaryConfusionMatrix=binaryconfMat, binary=binary)

        if dataset == 'Test':
            Y_test = Y_cat_format(self.testDf, self.YVar, binary)
            XY_test = self.testDf[self.XVars]
            XY_test['Y_test'] = Y_test
            XY_test.dropna(inplace=True)
            X_test = XY_test[self.XVars]
            Y_test = XY_test['Y_test']

            print(f'Train length: {len(self.Y_train)} --- Test length:{len(Y_test)}')
            print(f'Train variance: {np.var(self.Y_train)} --- Test variance:{np.var(Y_test)}')

            accuracy = cross_val_score(self.clf, X_test, Y_test, cv=crossVals, scoring=scoring)

            y_test_predict = cross_val_predict(self.clf, X_test, Y_test, cv=crossVals)
            confMat = confusion_matrix(Y_test, y_test_predict)

            if binary:
                precision = precision_score(Y_test, y_test_predict)
                recall = recall_score(Y_test, y_test_predict)
                precisions, recalls, thresholds = precision_recall_curve(Y_test, y_test_predict)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'{Name}')
                plt.show()

                fpr, tpr, thresholds = roc_curve(Y_test, y_test_predict)
                plot_roc_curve(fpr, tpr, Name)
                plt.show()

                scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall,
                                           confusionMatrix=confMat,
                                           binaryConfusionMatrix='NA', binary=binary, roc_curve=None, best_params=None)

            else:
                binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                          [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
                precision = confMat[0, 0] / (confMat[0, 0] + confMat[1, 0])
                recall = confMat[0, 0] / (confMat[0, 0] + confMat[0, 1])

                scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall,
                                           confusionMatrix=confMat,
                                           binaryConfusionMatrix=binaryconfMat, binary=binary, best_params=None,
                                           roc_curve=None)

        if dataset == 'Validation':
            Y_val = Y_cat_format(self.valDf, self.YVar, binary)
            XY_val = self.valDf[self.XVars]
            XY_val['Y_val'] = Y_val
            XY_val.dropna(inplace=True)
            X_val = XY_val[self.XVars]
            Y_val = XY_val['Y_val']

            accuracy = cross_val_score(self.clf, X_val, Y_val, cv=crossVals, scoring=scoring)

            y_val_predict = cross_val_predict(self.clf, X_val, Y_val, cv=crossVals)
            confMat = confusion_matrix(Y_val, y_val_predict)

            if binary:
                precision = precision_score(Y_val, y_val_predict)
                recall = recall_score(Y_val, y_val_predict)
                precisions, recalls, thresholds = precision_recall_curve(Y_val, y_val_predict)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'{Name}')
                plt.show()

                fpr, tpr, thresholds = roc_curve(Y_val, y_val_predict)
                plot_roc_curve(fpr, tpr, Name)
                plt.show()

                scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall,
                                           confusionMatrix=confMat, binaryConfusionMatrix='NA', binary=binary)

            else:
                binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                          [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
                precision = confMat[0, 0] / (confMat[0, 0] + confMat[1, 0])
                recall = confMat[0, 0] / (confMat[0, 0] + confMat[0, 1])
                scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall,
                                           confusionMatrix=confMat,
                                           binaryConfusionMatrix=binaryconfMat, binary=binary)

        return scoresObject

class TSClassifier:

    def __init__(self, full_df, startDate, XVars, YVar, binary=False, remove_duplicate_dates=False, n_splits=5):

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
            n_correct = sum(y_pred == Y_train_folds)
            trainAccuracy.append(n_correct / len(y_pred))
            confMat_current = confusion_matrix(Y_train_folds, y_pred)
            trainConfMat.append(confMat_current)

            if self.binary:
                trainPrecision.append(precision_score(Y_train_folds, y_pred))
                trainRecall.append(recall_score(Y_train_folds, y_pred))
                precisions, recalls, thresholds = precision_recall_curve(Y_train_folds, y_pred)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'Train')
                plt.show()

                fpr, tpr, thresholds = roc_curve(Y_train_folds, y_pred)
                plot_roc_curve(fpr, tpr, 'Train')
                plt.show()


            else:
                binaryconfMat_current = np.array(
                    [[np.sum(confMat_current[0:4, 0:4]), np.sum(confMat_current[0:4, 4:8])],
                     [np.sum(confMat_current[4:8, 0:4]), np.sum(confMat_current[4:8, 4:8])]])

                trainBinaryConfMat.append(binaryconfMat_current)
                trainPrecision.append(confMat_current[0, 0] / (confMat_current[0, 0] + confMat_current[1, 0]))
                trainRecall.append(confMat_current[0, 0] / (confMat_current[0, 0] + confMat_current[0, 1]))

        trainingScores = scoreHolder(accuracy=trainAccuracy, precision=trainPrecision, recall=trainRecall,
                                   confusionMatrix=trainConfMat,
                                   binaryConfusionMatrix=trainBinaryConfMat, binary=self.binary, best_params=None,
                                   roc_curve=None)

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

            print(f'Train length: {len(Y_train_folds)} --- Test length:{len(Y_test_folds)}')
            print(f'Train variance: {np.var(Y_train_folds)} --- Test variance:{np.var(Y_test_folds)}')

            clone_clf.fit(X_train_folds, Y_train_folds)
            y_pred = clone_clf.predict(X_test_folds)
            n_correct = sum(y_pred == Y_test_folds)
            testAccuracy.append(n_correct / len(y_pred))
            confMat_current = confusion_matrix(Y_test_folds, y_pred)
            testConfMat.append(confMat_current)

            if self.binary:
                testPrecision.append(precision_score(Y_test_folds, y_pred))
                testRecall.append(recall_score(Y_test_folds, y_pred))
                precisions, recalls, thresholds = precision_recall_curve(Y_test_folds, y_pred)
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
                plt.title(f'Test')
                plt.show()

                fpr, tpr, thresholds = roc_curve(Y_test_folds, y_pred)
                plot_roc_curve(fpr, tpr, 'Test')
                plt.show()


            else:
                binaryconfMat_current = np.array(
                    [[np.sum(confMat_current[0:4, 0:4]), np.sum(confMat_current[0:4, 4:8])],
                     [np.sum(confMat_current[4:8, 0:4]), np.sum(confMat_current[4:8, 4:8])]])

                testBinaryConfMat.append(binaryconfMat_current)
                testPrecision.append(confMat_current[0, 0] / (confMat_current[0, 0] + confMat_current[1, 0]))
                testRecall.append(confMat_current[0, 0] / (confMat_current[0, 0] + confMat_current[0, 1]))

        if len(testBinaryConfMat) == 0: testBinaryConfMat = 'NA'


        testScores = scoreHolder(accuracy=testAccuracy, precision=testPrecision, recall=testRecall,
                                   confusionMatrix=testConfMat,
                                   binaryConfusionMatrix=testBinaryConfMat, binary=self.binary, best_params=None,
                                   roc_curve=None)

        return trainingScores, testScores

class GridSearch:

    def __init__(self, full_df, clf, startDate, remove_duplicate_dates):

        if 'Unnamed: 0' in full_df.columns:
            full_df.drop('Unnamed: 0', axis=1, inplace=True)

        self.trainDf, self.testDf, self.valDf = dataSetup(full_df, startDate, remove_duplicate_dates=remove_duplicate_dates)
        self.clf = clf
        self.startDate = startDate
        self.full_df=full_df

    def runGrid_search(self, XVars, YVar, param_grid, split_type, remove_duplicate_dates, n_splits=5,
                       binary=False, n_jobs=-1):

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

            print(f'Cross sectional parameter analysis:')
            print(f'Train length: {len(Y_train)} --- Test length:{len(Y_test)}')
            print(f'Train variance: {np.var(Y_train)} --- Test variance:{np.var(Y_test)}')

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

        print(f'Cross-section parameter analysis:')
        print("Param for GS", bestParams)
        print("CV score for GS", bestScore)
        print("Train AUC ROC Score for GS: ", train_roc_auc_score)
        print("Test AUC ROC Score for GS: ", roc_auc_score(Y_test, grid_cv.predict(X_test)))

        train_scores = scoreHolder(best_params=bestParams, accuracy=bestScore, roc_auc_score=train_roc_auc_score)

        return train_scores


class regressor:

    def __init__(self):
        print('Not ready yet')

def runML_tests(full_df,startDate,XVars, YVar,crossVals,scoring,ML_type,remove_duplicate_dates,n_splits = 5, clf_type=None,
                reg_type=None,GS_clf_type=None ,binary=True,random_state=42, GS_params=None
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

            train_scores = classifier.testCSClassifier(Name='Train', crossVals=crossVals, scoring=scoring,
                                                       dataset='Train')
            test_scores = classifier.testCSClassifier(Name='Test', crossVals=crossVals, scoring=scoring, dataset='Test')
            val_scores = classifier.testCSClassifier(Name='Validate', crossVals=crossVals, scoring=scoring,
                                                     dataset='Validate')

        else:
            classifier = TSClassifier(full_df=full_df, startDate=startDate, XVars=XVars, YVar=YVar, binary=binary,
                                      remove_duplicate_dates=remove_duplicate_dates, n_splits=n_splits)

            train_scores, test_scores = classifier.testTSClassifier(clf=clf)
            val_scores = None

        print(
                f'Training Scores:\n Accuracy: {train_scores.accuracy}\n'
                f' Precision: {train_scores.precision}\n Recall: {train_scores.recall}\n'
                f' Confusion Matrix:\n{train_scores.confusionMatrix}\n'
                f' Binary Confusion Matrix:\n{train_scores.binaryConfusionMatrix}\n\n')

        print(
                f'Training Scores:\n Accuracy: {test_scores.accuracy}\n'
                f' Precision: {test_scores.precision}\n Recall: {test_scores.recall}\n'
                f' Confusion Matrix:\n{test_scores.confusionMatrix}\n'
                f' Binary Confusion Matrix:\n{test_scores.binaryConfusionMatrix}\n\n')

        return train_scores, test_scores, val_scores

    if ML_type in ['CS_Gridsearch', 'TS_Gridsearch']:

        GS_clf = setGridsearch(GS_params=GS_params,GS_clf_type=GS_clf_type,random_state=random_state)

        if ML_type == 'CS_Gridsearch': split_type='CS'
        else: split_type='TS'


        if not binary == True:
            raise ValueError("Grid search currently requires binary input")

        GS_object = GridSearch(full_df=full_df,clf=GS_clf,startDate=startDate,
                                   remove_duplicate_dates=remove_duplicate_dates)

        GS_scores = GS_object.runGrid_search(XVars=XVars,YVar=YVar,
                remove_duplicate_dates=remove_duplicate_dates, param_grid=GS_params,
                split_type=split_type, binary=binary, n_jobs=-1)

        return GS_scores



    if ML_type in ['CS_Regressor','TS_Regressor']:

        print("Regressor class not active yet")

        if ML_type == 'CS_Regressor':
            print()

        if ML_type == 'TS_Regressor':
            print()




