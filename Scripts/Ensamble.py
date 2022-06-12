import pandas as pd
from ML_Tests import runML_tests, setupXYvars
from ML_stacking import checkScore
import numpy as np


def runRegLoops():
    dateList = []
    rdList = []
    Reg_TypeList = []
    reg_algoList = []
    datasetsList = []
    trainMSEList = []
    trainMAEList = []
    testMSEList = []
    testMAEList = []
    valMSEList = []
    valMAEList = []

    date = '1950-01-01'
    rd = False
    Reg_Type = 'CS_Regressor'
    for algo in ['reg_SGD','reg_NN']:
        print(f"{date}\n"
              f"duplicates removed: {rd}\n"
              f"Classification type: {Reg_Type}:"
              f"Classification algo: {algo}\n"
              f"Dataset: sameday_boosted")

        trainScores, testScores, valScores = runML_tests(full_df=df, XVars=sameday_boosted, YVar=Y_var,
                                                         remove_duplicate_dates=rd,
                                                         crossVals=5, scoring='accuracy', reg_type=algo,
                                                         ML_type=Reg_Type,
                                                         startDate=date,return_prediction=False, binary=False)

        trainMSE = checkScore(trainScores.MSE)
        trainMAE = checkScore(trainScores.MAE)

        testMSE = checkScore(testScores.MSE)
        testMAE = checkScore(testScores.MAE)

        valMSE = checkScore(valScores.MSE)
        valMAE = checkScore(valScores.MAE)

        print(f"Train Scores: MSE: {trainMSE} MAE: {trainMAE}\n\n"
              f"Test Scores MSE: {testMSE} MAE: {testMAE}\n\n"
              f"Val Scores MSE: {valMSE} MAE: {valMAE}\n\n")

        dateList.append(date)
        rdList.append(rd)
        Reg_TypeList.append(Reg_Type)
        reg_algoList.append(algo)
        datasetsList.append('sameday_boosted')

        trainMSEList.append(trainMSE)
        trainMAEList.append(trainMAE)

        testMSEList.append(testMSE)
        testMAEList.append(testMAE)

        valMSEList.append(valMSE)
        valMAEList.append(valMAE)

    listDict={'Dates':dateList,'Duplicates_removed':rdList,'Reg_type':Reg_TypeList,
              'Algo':reg_algoList,'XVars':datasetsList,'Train_MSE':trainMSEList,'Train_MAE':trainMAEList,
              'Test_MSE':testMSEList,'Test_MAE':testMAEList,'Val_MSE':valMSEList,
                'Val_MAE':valMAEList}


    regScores_df = pd.DataFrame(listDict)

    # noinspection PyTypeChecker
    regScores_df.to_csv(f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Combine_sameday_boosted_CS_Regression{regScores_df.shape}.csv')

def chopLists(dict):
    shortest = 10000000
    for i in dict:
        if len(dict[i]) < shortest: shortest = len(dict[i])

    for i in dict:
        if len(dict[i]) != shortest:
            for j in range(shortest, len(dict[i])):
                del dict[i][-1]

    return dict

def setup():
    df = pd.read_csv(
        '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final_transposed.csv')

    for i in df.columns:
        if 'Unnamed' in i:
            df.drop(columns=i, inplace=True)


    train_0, test_0, val_0, X_train, Y_train, X_test, Y_test, X_val, Y_val =  runML_tests(full_df=df, XVars=X_control_vars + PVDBOW + PVDM, YVar=Y_var,
                                                           remove_duplicate_dates=False,
                                                           crossVals=5, scoring='accuracy', reg_type='reg_NN',
                                                           ML_type='CS_Regressor',
                                                           startDate='1950-01-01', return_prediction=True, binary=False)

    train_1, test_1, val_1, _, _, _, _, _, _ = runML_tests(full_df=df, XVars=X_control_vars + PVDM, YVar=Y_var,
                                                           remove_duplicate_dates=False,
                                                           crossVals=5, scoring='accuracy', reg_type='reg_NN',
                                                           ML_type='CS_Regressor',
                                                           startDate='1950-01-01', return_prediction=True, binary=False)

    train_2, test_2, val_2, _, _, _, _, _, _ = runML_tests(full_df=df, XVars=X_control_vars, YVar=Y_var,
                                                           remove_duplicate_dates=False,
                                                           crossVals=5, scoring='accuracy', reg_type='reg_NN',
                                                           ML_type='CS_Regressor',
                                                           startDate='1950-01-01', return_prediction=True, binary=False)

    train_3, test_3, val_3, _, _, _, _, _, _ = runML_tests(full_df=df, XVars=X_control_vars + PVDBOW, YVar=Y_var,
                                                           remove_duplicate_dates=False,
                                                           crossVals=5, scoring='accuracy', reg_type='reg_NN',
                                                           ML_type='CS_Regressor',
                                                           startDate='1950-01-01', return_prediction=True, binary=False)

    train_4, test_4, val_4, _, _, _, _, _, _ = runML_tests(full_df=df, XVars=X_control_vars, YVar=Y_var,
                                                           remove_duplicate_dates=False,
                                                           crossVals=5, scoring='accuracy', reg_type='reg_MLR',
                                                           ML_type='CS_Regressor',
                                                           startDate='1950-01-01', return_prediction=True, binary=False)

    train_5, test_5, val_5, _, _, _, _, _, _ = runML_tests(full_df=df, XVars=PVDBOW,YVar=Y_var,
                                                           remove_duplicate_dates=False, crossVals=5,scoring='accuracy',
                                                           reg_type='reg_SGD', ML_type='CS_Regressor',
                                                           startDate='1950-01-01', return_prediction=True,binary=False)



    train_X_df = pd.DataFrame(data=X_train)
    train_dict = {'logDif_date_resid': Y_train.tolist(), 'pred_0': train_0.tolist(), 'pred_1': train_1.tolist(),
                  'pred_2': train_2.tolist().pop(), 'pred_3': train_3.tolist(), 'pred_4': train_4.tolist().pop(),
                  'pred_5': train_5.tolist().pop()}

    train_pred_df = pd.DataFrame(data=train_dict)
    train_df = pd.concat([train_X_df, train_pred_df], axis=1)

    test_X_df = pd.DataFrame(data=X_test)
    test_dict = {'logDif_date_resid': Y_test.tolist(), 'pred_0': test_0.tolist(), 'pred_1': test_1.tolist(),
              'pred_2': test_2.tolist(), 'pred_3': test_3.tolist(),
              'pred_4': test_4.tolist(), 'pred_5': test_5.tolist().pop()}

    test_pred_df = pd.DataFrame(
        data=test_dict)
    test_df = pd.concat([test_X_df, test_pred_df], axis=1)

    val_X_df = pd.DataFrame(data=X_val)
    val_dict = {'logDif_date_resid': Y_val.tolist(), 'pred_0': val_0.tolist(), 'pred_1': val_1.tolist(),
              'pred_2': val_2.tolist(), 'pred_3': val_3.tolist(),
              'pred_4': val_4.tolist(), 'pred_5': val_5.tolist()}

    val_pred_df = pd.DataFrame(
        data=val_dict)
    val_df = pd.concat([val_X_df, val_pred_df], axis=1)

    df = pd.concat([train_df, test_df, val_df], ignore_index=True, keys=['train', 'test', 'val'])

    Dates = []
    for i in range(len(df)):
        Dates.append('1951-01-01')

    df['Date'] = Dates

    # noinspection PyTypeChecker
    df.to_csv(
        '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final_boosted.csv')

    return df

PVDM = ['Days_since_last_speech']
for i in range(0, 20):
    PVDM.append((f'PVDM_{i}'))

PVDBOW = ['Days_since_last_speech']
for i in range(0, 20):
    PVDBOW.append((f'PVDBOW_{i}'))

X_control_vars, X_meta_vars, X_test_vars, Y_var = setupXYvars(['PVDBOW', 'PVDM'])
X_preds = ['pred_0','pred_1','pred_2','pred_3','pred_4','pred_5']

sameday_boosted = X_control_vars+PVDM+PVDBOW+X_preds

df = setup()

#df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final_boosted.csv')

for i in df.columns: print(i)

runRegLoops()

