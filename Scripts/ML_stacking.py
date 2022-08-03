import pandas as pd

from ML_Tests import runML_tests, dataSetup, setupXYvars
from Feature_selection import Shrinkage_Methods

def setupSingleRun():
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

    # GS_clf = GStree_clf  # empty classifiers for grid search. Options: GSForrest_clf, GSNN_clf, GStree_clf


filePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Combine_sameday_speeches/Combine_sameday_speeches_final_transposed.csv'

nonCombineddf = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(73827, 458).csv')

df = pd.read_csv(filePath)
print(f'There are {len(df)} entries in the df and {len(df["Date"].unique())} unique dates in the df for a ratio of {len(df["Date"].unique())/len(df)}')

for i in df.columns:
    if 'Unnamed' in i:
        df.drop(columns=i,inplace=True)



X_control, X_meta, X_test, Y = setupXYvars(['PVDBOW','PVDM']) #options: ['WV','DV_200_', 'DV_20_','vader',blob']

possibleBestVars = ['DlogDif_1', 'DlogDif_2', 'pos_neg_transform','Nasdaq_ld_1', 'Oil_ld_1','VIX_ld_1','DV_20_6','DV_20_8','DV_20_13','DV_20_15']

Datasets = {'Auto': X_control , 'Meta': X_meta , 'NLP': X_test ,
            'AutoMeta': X_control + X_meta , 'AutoNLP': X_control+X_test , 'MetaNLP':X_meta+X_test ,
            'All': X_control + X_meta + X_test,'PossibleBest':possibleBestVars}

vader = ['VaderNeg', 'VaderNeu', 'VaderPos', 'VaderComp']

blob = ['blobPol', 'blobSubj']

WV = []
for i in range(0, 200):
    WV.append(f'WV_{i}')

DV_200 = []
for i in range(0, 200):
    DV_200.append(f'DV_200_{i}')

DV_20 = []
for i in range(0, 20):
    DV_20.append(f'DV_20_{i}')

PVDM=['Days_since_last_speech']
for i in range(0,20):
    PVDM.append((f'PVDM_{i}'))

PVDBOW=['Days_since_last_speech']
for i in range(0,20):
    PVDBOW.append((f'PVDBOW_{i}'))

NLPDatasets = {'Vader':vader, 'Blob':blob, 'WV':WV, 'DV_200':DV_200, 'DV_20':DV_20}

combine_sameday_datasets = {'Auto':X_control,'AutoBoth':X_control+PVDM+PVDBOW,'AutoPVDM':X_control+PVDM,'AutoPVDBOW':X_control+PVDBOW
                            ,'PVDM':PVDM, 'PVDBOW':PVDBOW,
                            'BothSameDaySets':PVDM+PVDBOW}

XVars = {'Auto':X_control ,'AutoPVDBOW':X_control+PVDBOW, 'PVDBOW':PVDBOW} #,'AutoPVDBOW':X_control+PVDBOW

Clf_Types = ['TS_Classifier'] #'CS_Classifier',

Reg_Types = ['CS_Regressor'] #,'TS_Regressor'

StartDates = ['1950-01-01']#'1998-01-01','2000-01-01',  #'1990-01-01' - Some meta data does not date back far enough to begin before 1998

Binary = [True] #True,False

Remove_duplicates = [False] #True, False

reg_algos = ['reg_NN', 'reg_SGD']#'reg_GradientBoosting','reg_MLR','reg_SGD'

clf_algos = ['clf_logreg','clf_SGD'] #,'clf_KNN',,'clf_GradientBoosting','clf_NN'

def runClfLoops():
    dateList = []
    rdList = []
    binaryList = []
    Clf_TypeList = []
    clf_algoList = []
    datasetsList = []
    trainAccuracyList = []
    trainPrecisionList = []
    trainRecallList = []
    testAccuracyList = []
    testPrecisionList = []
    testRecallList = []
    valAccuracyList = []
    valPrecisionList = []
    valRecallList = []

    for date in StartDates:

        for rd in Remove_duplicates:

            for binary in Binary:

                for Clf_Type in Clf_Types:

                    for algo in clf_algos:

                        for X in XVars:

                            print(f"{date}\n"
                                  f"duplicates removed: {rd}\n"
                                  f"binary: {binary}\n"
                                  f"Classification type: {Clf_Type}:"
                                  f"Classification algo: {algo}\n"
                                  f"Dataset: {X}")

                            trainScores, testScores, valScores = runML_tests(full_df=df, XVars=XVars[X], YVar=Y ,
                                        remove_duplicate_dates=rd,
                                        crossVals=5, scoring='accuracy', clf_type=algo, ML_type=Clf_Type,
                                        binary=binary,return_prediction=False,startDate=date)

                            trainScoreAcc = checkScore(trainScores.accuracy)
                            trainScorePrec = checkScore(trainScores.precision)
                            trainScoreRec = checkScore(trainScores.recall)

                            testScoreAcc = checkScore(testScores.accuracy)
                            testScorePrec = checkScore(testScores.precision)
                            testScoreRec = checkScore(testScores.recall)

                            valScoreAcc = checkScore(valScores.accuracy)
                            valScorePrec = checkScore(valScores.precision)
                            valScoreRec = checkScore(valScores.recall)

                            print(f"Train Scores: Accuracy: {trainScoreAcc} Precision: {trainScorePrec} Recall: {trainScoreRec}\n\n"
                                  f"Test Scores Accuracy: {testScoreAcc} Precision: {testScorePrec} Recall: {testScoreRec}\n\n"
                                  f"Val Scores Accuracy: {valScoreAcc} Precision: {valScorePrec} Recall: {valScoreRec}\n\n")

                            dateList.append(date)
                            rdList.append(rd)
                            binaryList.append(binary)
                            Clf_TypeList.append(Clf_Type)
                            clf_algoList.append(algo)
                            datasetsList.append(X)

                            trainAccuracyList.append(trainScoreAcc)
                            trainPrecisionList.append(trainScorePrec)
                            trainRecallList.append(trainScoreRec)

                            testAccuracyList.append(testScoreAcc)
                            testPrecisionList.append(testScorePrec)
                            testRecallList.append(testScoreRec)

                            valAccuracyList.append(valScoreAcc)
                            valPrecisionList.append(valScorePrec)
                            valRecallList.append(valScoreRec)


    listDict={'Dates':dateList,'Duplicates_removed':rdList,'Binary':binaryList,'Clf_type':Clf_TypeList,
              'Algo':clf_algoList,'XVars':datasetsList,'Train_acc':trainAccuracyList,'Train_prec':trainPrecisionList,
              'Train_recall':trainRecallList,'Test_acc':testAccuracyList,'Test_prec':testPrecisionList,
              'Test_recall':testRecallList,'Val_acc':valAccuracyList,'Val_prec':valPrecisionList,
                'Val_recall':valRecallList}


    clfScores_df = pd.DataFrame(listDict)

    # noinspection PyTypeChecker
    clfScores_df.to_csv(f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Sameday_1950_binary_3{clfScores_df.shape}.csv')

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

    for date in StartDates:

        for rd in Remove_duplicates:

            for Reg_Type in Reg_Types:

                for algo in reg_algos:

                    for X in XVars:

                        print(f"{date}\n"
                                f"duplicates removed: {rd}\n"
                                f"Classification type: {Reg_Type}:"
                                f"Classification algo: {algo}\n"
                                f"Dataset: {X}")

                        trainScores, testScores, valScores = runML_tests(full_df=df, XVars=XVars[X], YVar=Y ,
                                    remove_duplicate_dates=rd,
                                    crossVals=5, scoring='accuracy', reg_type=algo, ML_type=Reg_Type,
                                    startDate=date,return_prediction=False,binary=False)

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
                        datasetsList.append(X)

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
    regScores_df.to_csv(f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Combine_sameday_CS_Regression_2{regScores_df.shape}.csv')

def checkScore(score):
    try: finalScore = round(max(score),3)
    except:
        try: finalScore = round(score,3)
        except: finalScore = score

    return finalScore

figSavePath = f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Scripts/Solution/19119461/Thesis_WriteUp/Images/ELasticGrid_DV_20_3_0,1.png'

X_1 = X_control + X_meta + DV_20
X_2 = ['DlogDif_1', 'DlogDif_2', 'Nasdaq_ld_1', 'blackSwan_SD3_1','pos_neg_transform','USDX_ld_1']
X_3 = ['DlogDif_1', 'DlogDif_2', 'Nasdaq_ld_1','pos_neg_transform','SSE_ld_1','VIX_ld_1','Oil_ld_1','USDX_ld_1','blackSwan_SD4_1','stdVol_1DateResid']
possBest2 = ['DlogDif_1', 'DlogDif_2', 'Nasdaq_ld_1','pos_neg_transform','Oil_ld_1','VIX_ld_1','DV_20_6','DV_20_8','DV_20_13','DV_20_15']
for i in possBest2:
    X_1.remove(i)

shrinkTest = Shrinkage_Methods( nonCombineddf, DV_20 ,'DlogDif', 10)

shrinkTest.Elastic_Gridsearch(l1_ratio=0.1, show_coefficients=True, minAlpha=0, maxAlpha=2, figSavePath=figSavePath)







