import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt

def splitTestTrainVal(df):
    testDf = df.sample(frac = 0.15,random_state = 25)
    df = df.drop(testDf.index)
    valDf = df.sample(frac = 0.1,random_state = 25)
    trainDf = df.drop(valDf.index)

    return trainDf,testDf,valDf

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

def runSGD(full_df, startDate,XVars,YVar,crossVals=3,scoring='accuracy',binary=False, dataset = 'Train'):
    if 'Unnamed: 0' in full_df.columns:
        full_df.drop('Unnamed: 0', axis=1, inplace=True)

    small_Df = full_df[~(full_df['Date'] < startDate)]
    small_Df.reset_index(inplace=True)

    trainDf, testDf, valDf = splitTestTrainVal(small_Df)

    Y_train = Y_cat_format(trainDf,YVar,binary)
    XY_train = trainDf[XVars]
    XY_train['Y_train'] = Y_train
    XY_train.dropna(inplace=True)
    X_train = XY_train[XVars]
    Y_train = XY_train['Y_train']

    sgd_clf = SGDClassifier(random_state=42, shuffle=False,loss='log',max_iter=10000)
    sgd_clf.fit(X_train, Y_train)

    if dataset == 'Train':

        accuracy = cross_val_score(sgd_clf, X_train, Y_train, cv=crossVals, scoring=scoring)

        y_train_predict = cross_val_predict(sgd_clf, X_train, Y_train, cv=crossVals)
        confMat = confusion_matrix(Y_train, y_train_predict)

        if binary:
            precision = precision_score(Y_train,y_train_predict)
            recall = recall_score(Y_train,y_train_predict)
            return np.max(accuracy), precision, recall, confMat

        return accuracy, 'NA', 'NA', confMat

    if dataset == 'Test':
        Y_test = Y_cat_format(testDf, YVar, binary)
        XY_test = testDf[XVars]
        XY_test['Y_test'] = Y_test
        XY_test.dropna(inplace=True)
        X_test = XY_test[XVars]
        Y_test = XY_test['Y_test']

        accuracy = cross_val_score(sgd_clf, X_test, Y_test, cv=crossVals, scoring=scoring)

        y_test_predict = cross_val_predict(sgd_clf, X_test, Y_test, cv=crossVals)
        confMat = confusion_matrix(Y_test, y_test_predict)

        if binary:
            precision = precision_score(Y_test,y_test_predict)
            recall = recall_score(Y_test,y_test_predict)
            return np.max(accuracy), precision, recall, confMat

        return accuracy, 'NA', 'NA', confMat

    if dataset == 'Validation':
        Y_val = Y_cat_format(valDf, YVar, binary)
        XY_val = valDf[XVars]
        XY_val['Y_val'] = Y_val
        XY_val.dropna(inplace=True)
        X_val = XY_val[XVars]
        Y_val = XY_val['Y_val']

        accuracy = cross_val_score(sgd_clf, X_val, Y_val, cv=crossVals, scoring=scoring)

        y_val_predict = cross_val_predict(sgd_clf, X_val, Y_val, cv=crossVals)
        confMat = confusion_matrix(Y_val, y_val_predict)

        if binary:
            precision = precision_score(Y_val,y_val_predict)
            recall = recall_score(Y_val,y_val_predict)
            return np.max(accuracy), precision, recall, confMat

        return accuracy, 'NA', 'NA', confMat

full_df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(106774, 237).csv')
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values(by='Date', inplace=True)

vars = []
for i in range(0, 199):
    vars.append(f'V_{i}')

X_auto = ['DlogDif_1', 'DlogDif_2', 'absDlogDif_1', 'blackSwan_SD3_1', 'blackSwan_SD4_1', 'blackSwan_SD5_1',
              'stdVol_1DateResid', 'pos_neg_transform']
X_NLP = ['VaderNeg', 'VaderNeu', 'VaderPos', 'VaderComp', 'blobPol', 'blobSubj'] + vars
X_meta = ['BTC_ld_1', 'BTC_dr_1', 'Nasdaq_ld_1', 'Nasdaq_dr_1', 'Oil_ld_1', 'Oil_dr_1', 'SSE_ld_1', 'SSE_dr_1',
              'USDX_ld_1', 'USDX_dr_1', 'VIX_ld_1', 'VIX_dr_1']

Y = ['DlogDif', 'logDif', 'logDif_date_resid']
Y = ['logDif']

Y_auto_accs = []
Y_NLP_accs = []
Y_meta_accs = []
Y_autoNLP_accs = []
Y_autometa_accs = []
Y_all_accs = []

startDate = '2010-01-01'
crossVals = 3
scoring = 'accuracy' #'accuracy'
binary = False
dataset = 'Test' #Options: Train, Test, Validation

for i in Y:
    Y_auto_accs = runSGD(full_df,startDate,X_auto,i,crossVals=crossVals,scoring=scoring,binary=binary,dataset=dataset)
    Y_NLP_accs = runSGD(full_df,startDate,X_NLP,i,crossVals=crossVals,scoring=scoring,binary=binary,dataset=dataset)
    Y_meta_accs = runSGD(full_df,startDate,X_meta,i,crossVals=crossVals,scoring=scoring,binary=binary,dataset=dataset)
    Y_autoNLP_accs = runSGD(full_df,startDate,X_auto+X_NLP,i,crossVals=crossVals,scoring=scoring,binary=binary,dataset=dataset)
    Y_autometa_accs = runSGD(full_df,startDate,X_auto+X_meta,i,crossVals=crossVals,scoring=scoring,binary=binary,dataset=dataset)
    Y_all_accs = runSGD(full_df,startDate,X_auto+X_NLP+X_meta,i,crossVals=crossVals,scoring=scoring,binary=binary,dataset=dataset)


print(f'{dataset} dataset performance:\n______________________________________________________________\n\n'
      f'____________Auto_____________:\n Accuracy: {Y_auto_accs[0]}\n Precision: {Y_auto_accs[1]}\n Recall: {Y_auto_accs[2]}\n Confusion Matrix:\n{Y_auto_accs[3]}\n\n'
      f'____________NLP_____________:\n Accuracy: {Y_NLP_accs[0]}\n Precision: {Y_NLP_accs[1]}\n Recall: {Y_NLP_accs[2]}\n Confusion Matrix:\n{Y_NLP_accs[3]}\n\n'
      f'____________Meta_____________:\n Accuracy: {Y_meta_accs[0]}\n Precision: {Y_meta_accs[1]}\n Recall: {Y_meta_accs[2]}\n Confusion Matrix:\n{Y_meta_accs[3]}\n\n'
      f'____________AutoNLP_____________:\n Accuracy: {Y_autoNLP_accs[0]}\n Precision: {Y_autoNLP_accs[1]}\n Recall: {Y_autoNLP_accs[2]}\n Confusion Matrix:\n{Y_autoNLP_accs[3]}\n\n'
      f'____________AutoMeta_____________:\n Accuracy: {Y_autometa_accs[0]}\n Precision: {Y_autometa_accs[1]}\n Recall: {Y_autometa_accs[2]}\n Confusion Matrix:\n{Y_autometa_accs[3]}\n\n'
      f'____________All_____________:\n Accuracy: {Y_all_accs[0]}\n Precision: {Y_all_accs[1]}\n Recall: {Y_all_accs[2]}\n Confusion Matrix:\n{Y_all_accs[3]}\n\n')
