import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
plt.style.use('seaborn')


class scoreHolder:
    def __init__(self,accuracy,precision,recall,confusionMatrix,binaryConfusionMatrix ,roc_curve, best_params, binary):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.confusionMatrix = confusionMatrix
        self.binaryConfusionMatrix = binaryConfusionMatrix
        self.roc_curve = roc_curve
        self.best_params = best_params
        self.binary = binary

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

def dataSetup(full_df,startDate):
    if 'Unnamed: 0' in full_df.columns:
        full_df.drop('Unnamed: 0', axis=1, inplace=True)

    small_Df = full_df[~(full_df['Date'] < startDate)]
    small_Df.reset_index(inplace=True)
    small_Df.sort_values(by='Date')

    trainDf, testDf = train_test_split(small_Df,train_size=0.7,random_state=42)
    testDf, valDf = train_test_split(testDf,train_size=0.5,random_state=42) #shuffling is fine because YVar is stationary

    return trainDf, testDf, valDf

def runTSClassifier(Name,full_df,startDate,clf,XVars,YVar,binary = False,dataset = 'Train', n_splits=5):

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

def runCSClassifier(Name,clf, full_df, startDate,XVars,YVar,crossVals=3,scoring='accuracy',binary=False, dataset = 'Train'):


    #might need to implement SKF here

    trainDf, testDf, valDf = dataSetup(full_df,startDate)

    Y_train = Y_cat_format(trainDf, YVar, binary)
    XY_train = trainDf[XVars]
    XY_train['Y_train'] = Y_train
    XY_train.dropna(inplace=True)
    X_train = XY_train[XVars]
    Y_train = XY_train['Y_train']

    clf.fit(X_train, Y_train)

    if dataset == 'Train':

        # might need to implement SKF here
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

            scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall, confusionMatrix=confMat, binaryConfusionMatrix='NA',binary=binary)


        else:
            binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                      [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
            precision = confMat[0,0]/(confMat[0,0] + confMat[1,0] )
            recall = confMat[0,0]/(confMat[0,0]+confMat[0,1])

            scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall, confusionMatrix=confMat,
                                   binaryConfusionMatrix=binaryconfMat, binary=binary)

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

            scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall, confusionMatrix=confMat, binaryConfusionMatrix='NA',binary=binary)

        else:
            binaryconfMat = np.array([[np.sum(confMat[0:4, 0:4]), np.sum(confMat[0:4, 4:8])],
                                      [np.sum(confMat[4:8, 0:4]), np.sum(confMat[4:8, 4:8])]])
            precision = confMat[0,0]/(confMat[0,0] + confMat[1,0] )
            recall = confMat[0,0]/(confMat[0,0]+confMat[0,1])

            scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall, confusionMatrix=confMat,
                                   binaryConfusionMatrix=binaryconfMat, binary=binary)

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

            scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall, confusionMatrix=confMat, binaryConfusionMatrix='NA',binary=binary)

        else:
            binaryconfMat = np.array([[np.sum(confMat[0:4,0:4]),np.sum(confMat[0:4,4:8])],[np.sum(confMat[4:8,0:4]),np.sum(confMat[4:8,4:8])]])
            precision = confMat[0,0]/(confMat[0,0] + confMat[1,0] )
            recall = confMat[0,0]/(confMat[0,0]+confMat[0,1])
            scoresObject = scoreHolder(accuracy=accuracy, precision=precision, recall=recall, confusionMatrix=confMat,
                                   binaryConfusionMatrix=binaryconfMat, binary=binary)

    return scoresObject

def runGrid_search(clf, full_df, startDate,XVars,YVar,param_grid,split_type,n_splits=5,binary=False,n_jobs=-1):
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

        small_Df_train = small_Df.head(round(len(small_Df) * ((n_splits-1)/n_splits)))
        X_train = np.array(small_Df_train[XVars])
        Y_prepped_train = Y_cat_format(small_Df_train, YVar, binary)
        Y_train = np.array(Y_prepped_train)

        small_Df_test = small_Df.tail(round(len(small_Df)/n_splits))
        X_test = np.array(small_Df_test[XVars])
        Y_prepped_test = Y_cat_format(small_Df_test, YVar, binary)
        Y_test = np.array(Y_prepped_test)

        #print(f'small_df = {len(small_Df)}\ntrain_df = {len(small_Df_train)}\ntest_df = {len(small_Df_test)}')

        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=n_jobs, cv=tscv, error_score='raise')

        grid_cv.fit(X_train,Y_train)

        print(f'Time series parameter analysis:')
        print("Param for GS", grid_cv.best_params_)
        print("CV score for GS", grid_cv.best_score_)
        print("Train AUC ROC Score for GS: ", roc_auc_score(Y_train, grid_cv.predict(X_train)))
        print("Test AUC ROC Score for GS: ", roc_auc_score(Y_test, grid_cv.predict(X_test)))

def runML_tests(filePath,startDate,X_control, X_meta, X_test, Y,crossVals,scoring,dataset,ML_type,clf_type=None,
                reg_type=None,GS_clf_type=None ,binary=True,random_state=42, GS_params=None
                ):

    full_df = pd.read_csv(filePath)
    if 'date' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['Date'])
        full_df.sort_values(by='date', inplace=True)
    else:
        if 'Date' in full_df.columns:
            full_df['Date'] = pd.to_datetime(full_df['Date'])
            full_df.sort_values(by='Date', inplace=True)
        else: raise ValueError(f'Dataframe at {filePath} must contain a "Date" or "date" column')


    if ML_type == 'CS_Classifier' or 'TS_Classifier':
        clf_types = ['clf_SGD','clf_MLP','clf_NN','clf_KNN','clf_logreg','clf_tree','clf_forrest']
        if not clf_type in clf_types:
            raise ValueError(f"Classifier ML_type chosen but non-classifier model chosen. Please choose from:\n{clf_types}")

        if clf_type == 'clf_SGD':
            clf = SGDClassifier(random_state=random_state, shuffle=False, loss='log', max_iter=10000)
        else:
            if clf_type == 'clf_MLP':
                clf = MLPClassifier(max_iter=1000, shuffle=False, learning_rate_init=0.01, learning_rate='adaptive',
                                    random_state=random_state)
            else:

                if clf_type == 'clf_NN':
                    inputVars = len(X_control + X_meta + X_test)
                    if inputVars / 5 < 8:
                        L_1 = 8
                    else:
                        L_1 = round(inputVars / 8)

                    if binary:
                        L_3 = 1
                    else:
                        L_3 = 7
                    clf = MLPClassifier(hidden_layer_sizes=(L_1, 8, L_3), activation='relu', solver='adam',
                                        max_iter=1000,
                                        random_state=random_state)
                else:
                    if clf_type == 'clf_KNN':
                        clf = KNeighborsClassifier(weights='distance', algorithm='auto', n_jobs=-1)
                    else:
                        if clf_type == 'clf_logreg':
                            clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=10000,
                                                     random_state=random_state)
                        else:
                            if clf_type == 'clf_tree':
                                clf = DecisionTreeClassifier(random_state=random_state)
                            else:
                                if clf_type == 'clf_forrest':
                                    clf = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                                                 max_samples=None, max_depth=5)

    if ML_type == 'CS_Gridsearch' or 'TS_Gridsearch':
        GS_clf_types = ['GSforrest', 'GSNN', 'GStree']

        if GS_params==None:
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

    if ML_type == 'CS_Classifier':

        for i in Y:
            Y_control_scores = runCSClassifier('Auto', clf, full_df, startDate, X_control, i, crossVals=crossVals,
                                            scoring=scoring,
                                            binary=binary, dataset=dataset)
            print(
                f'____________Auto_____________:\n Accuracy: {Y_control_scores[0]}\n Precision: {Y_control_scores[1]}\n Recall: {Y_control_scores[2]}\n Confusion Matrix:\n{Y_control_scores[3]}\n Binary Confusion Matrix:\n{Y_control_scores[4]}\n\n')

            Y_test_scores = runCSClassifier('NLP', clf, full_df, startDate, X_test, i, crossVals=crossVals,
                                           scoring=scoring,
                                           binary=binary, dataset=dataset)
            print(
                f'____________NLP_____________:\n Accuracy: {Y_test_scores[0]}\n Precision: {Y_test_scores[1]}\n Recall: {Y_test_scores[2]}\n Confusion Matrix:\n{Y_test_scores[3]}\nBinary Confusion Matrix:\n{Y_test_scores[4]}\n\n')

            Y_meta_scores = runCSClassifier('Meta', clf, full_df, startDate, X_meta, i, crossVals=crossVals,
                                            scoring=scoring,
                                            binary=binary, dataset=dataset)
            print(
                f'____________Meta_____________:\n Accuracy: {Y_meta_scores[0]}\n Precision: {Y_meta_scores[1]}\n Recall: {Y_meta_scores[2]}\n Confusion Matrix:\n{Y_meta_scores[3]}\nBinary Confusion Matrix:\n{Y_meta_scores[4]}\n\n')

            Y_controltest_scores = runCSClassifier('Auto-NLP', clf, full_df, startDate, X_control + X_test, i,
                                               crossVals=crossVals,
                                               scoring=scoring, binary=binary, dataset=dataset)
            print(
                f'____________AutoNLP_____________:\n Accuracy: {Y_controltest_scores[0]}\n Precision: {Y_controltest_scores[1]}\n Recall: {Y_controltest_scores[2]}\n Confusion Matrix:\n{Y_controltest_scores[3]}\nBinary Confusion Matrix:\n{Y_controltest_scores[4]}\n\n')

            Y_controlmeta_scores = runCSClassifier('Auto-Meta', clf, full_df, startDate, X_control + X_meta, i,
                                                crossVals=crossVals,
                                                scoring=scoring, binary=binary, dataset=dataset)
            print(
                f'____________AutoMeta_____________:\n Accuracy: {Y_controlmeta_scores[0]}\n Precision: {Y_controlmeta_scores[1]}\n Recall: {Y_controlmeta_scores[2]}\n Confusion Matrix:\n{Y_controlmeta_scores[3]}\nBinary Confusion Matrix:\n{Y_controlmeta_scores[4]}\n\n')

            Y_all_scores = runCSClassifier('All', clf, full_df, startDate, X_control + X_test + X_meta, i,
                                           crossVals=crossVals,
                                           scoring=scoring, binary=binary, dataset=dataset)
            print(
                f'____________All_____________:\n Accuracy: {Y_all_scores[0]}\n Precision: {Y_all_scores[1]}\n Recall: {Y_all_scores[2]}\n Confusion Matrix:\n{Y_all_scores[3]}\nBinary Confusion Matrix:\n{Y_all_scores[4]}\n\n')

    if ML_type == 'TS_Classifier':
        for i in Y:
            scores = runTSClassifier('AutoMeta', full_df, startDate, clf, X_control+ X_meta + X_test, i, binary=binary, dataset=dataset,
                                     n_splits=crossVals)

            for i in scores.columns:
                print(f'{i}:')
                for j in range(len(scores)):
                    print(f'{scores[i][j]}\n')
                print('\n')

    if ML_type == 'CS_Gridsearch':

        if not binary == True:
            raise ValueError("Grid search currently requires binary input")

        for i in Y:
            runGrid_search(clf=GS_clf, full_df=full_df, startDate=startDate, XVars=X_control+X_meta+X_test, YVar=i, param_grid=GS_params,
                           split_type='CS', binary=binary, n_jobs=-1)

    if ML_type == 'TS_Gridsearch':

        if not binary == True:
            raise ValueError("Grid search currently requires binary input")

        for i in Y:
            runGrid_search(clf=GS_clf, full_df=full_df, startDate=startDate, XVars=X_control+X_meta+X_test, YVar=i, param_grid=GS_params,
                           split_type='TS', binary=binary, n_jobs=-1)

    if ML_type == 'CS_Regressor':
        print()

    if ML_type == 'TS_Regressor':
        print()




