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


filePath = '/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(73827, 458).csv'

df = pd.read_csv(filePath)
print(f'There are {len(df)} entries in the df and {len(df["Date"].unique())} unique dates in the df for a ratio of {len(df["Date"].unique())/len(df)}')


#test = Shrinkage_Methods(data=df,X_variables=X_test,Y_variable=Y,num_features=12)
#print('\nLasso')
#Lasso = test.run_Lasso()
#print('\nElasticNet')
#ElasticNet = test.run_ElasticNet()
#print('\nRidge')
#Ridge = test.run_Ridge()
#trainDf,testDf,valDf = dataSetup(df,startDate='1990-01-01',remove_duplicate_dates=False)
#GS = Shrinkage_Methods(data=trainDf,X_variables=possibleBestVars,Y_variable=Y,num_features=12)
#GS.Elastic_Gridsearch(minAlpha=0,maxAlpha=10,l1_ratio=0.3,show_coefficients=True) #for lasso l1_ratio = 1, for ridge l1_ratio =0.011


X_control, X_meta, X_test, Y = setupXYvars(['DV_20_','vader','blob']) #options: ['WV','DV_200_', 'DV_20_','vader',blob']

possibleBestVars = ['DlogDif_1', 'DlogDif_2', 'pos_neg_transform','Nasdaq_ld_1', 'Oil_ld_1','VIX_ld_1','DV_20_6','DV_20_8','DV_20_13','DV_20_15']

Datasets = {'Auto': X_control , 'Meta': X_meta , 'NLP': X_test ,
            'AutoMeta': X_control + X_meta , 'AutoNLP': X_control+X_test , 'MetaNLP':X_meta+X_test ,
            'All': X_control + X_meta + X_test,'PossibleBest':possibleBestVars}

Clf_Types = ['CS_Classifier','TS_Classifier']

Reg_Types = ['CS_Regressor','TS_Regressor']

StartDates = ['1998-01-01','2000-01-01', '2010-01-01'] #'1990-01-01' - Some meta data does not date back far enough to begin before 1998

Binary = [False] #True,

Remove_duplicates = [True, False]

reg_algos = ['reg_GradientBoosting','reg_NN', 'reg_MLR','reg_SGD']

clf_algos = ['clf_GradientBoosting','clf_NN','clf_logreg','clf_SGD' ]

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

                        for X in Datasets:

                            print(f"{date}\n"
                                  f"duplicates removed: {rd}\n"
                                  f"binary: {binary}\n"
                                  f"Classification type: {Clf_Type}:"
                                  f"Classification algo: {algo}\n"
                                  f"Dataset: {X}")

                            trainScores, testScores, valScores = runML_tests(full_df=df, XVars=Datasets[X], YVar=Y ,
                                        remove_duplicate_dates=rd,
                                        crossVals=5, scoring='accuracy', clf_type=algo, ML_type=Clf_Type,
                                        binary=binary,startDate=date)

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
    clfScores_df.to_csv(f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Classification_results_resetYCats{clfScores_df.shape}.csv')

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

                    for X in Datasets:

                        print(f"{date}\n"
                                f"duplicates removed: {rd}\n"
                                f"Classification type: {Reg_Type}:"
                                f"Classification algo: {algo}\n"
                                f"Dataset: {X}")

                        trainScores, testScores, valScores = runML_tests(full_df=df, XVars=Datasets[X], YVar=Y ,
                                    remove_duplicate_dates=rd,
                                    crossVals=5, scoring='accuracy', reg_type=algo, ML_type=Reg_Type,
                                    startDate=date,binary=False)

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
    regScores_df.to_csv(f'/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Regression_results{regScores_df.shape}.csv')

def checkScore(score):
    try: finalScore = max(score)
    except: finalScore = score
    return finalScore

runClfLoops()
#runRegLoops()







