import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ML_Tests import Y_cat_format, setupXYvars


def setupReg(regResults):

    regResults = pd.melt(regResults,id_vars=['Dates', 'Duplicates_removed', 'ML_Type', 'Algo',
       'XVars'],value_vars = ['Train_MSE', 'Train_MAE', 'Test_MSE', 'Test_MAE', 'Val_MSE',
       'Val_MAE'])

    return regResults

def setupClf(clfResults):

    clsResults = pd.melt(clfResults,id_vars=['Dates', 'Duplicates_removed', 'Binary', 'ML_Type',
       'Algo', 'XVars'],value_vars = ['Train_acc', 'Train_prec', 'Train_recall', 'Test_acc',
       'Test_prec', 'Test_recall', 'Val_acc', 'Val_prec', 'Val_recall'])

    return clsResults

def getComparativeSubsets(data,results,scoreType, YVar,ML_Type, startDate:str, cat: bool=False, binary: bool = False):
    data = (data[~(data['Date'] < startDate)].dropna())

    if cat:
        data[YVar] = Y_cat_format(data, YVar, binary)

    Ydesc = data[YVar].describe()

    results = (results[~(results['Dates'] != startDate)].loc[results['variable'] == scoreType]
        .loc[results['ML_Type']==ML_Type])
    results = results['value']

    resultsDesc = results['value'].describe()

    return Ydesc,resultsDesc

def stdvsMeanMAE(data,results,scoreType, YVar, XVars, Algo, ML_Type, startDate:str, cat: bool=False, binary: bool = False):
    def compareStdMeanMae():
        Ydesc, Train_MAE_Desc = getComparativeSubsets(data=data, results=regResults, scoreType='Train_MAE',
                                                      YVar='logDif_date_resid'
                                                      , startDate='2010-01-01')
        _, Test_MAE_Desc = getComparativeSubsets(data=data, results=regResults, scoreType='Test_MAE',
                                                 YVar='logDif_date_resid'
                                                 , startDate='2010-01-01')
        _, Val_MAE_Desc = getComparativeSubsets(data=data, results=regResults, scoreType='Val_MAE',
                                                YVar='logDif_date_resid'
                                                , startDate='2010-01-01')


regResults = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Regression_results(384, 11).csv')
clfResults = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Results/Classification_results(768, 15).csv')
data = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(73827, 458).csv')

X_control, X_meta, X_test, Y = setupXYvars(['DV_20_','vader','blob'])

data = data[['Date',Y]+X_control + X_meta + X_test ]


print(max(data['Date']))

regResults['ML_Type'] = regResults['Reg_type']
regResults.drop('Reg_type',axis=1,inplace=True)

clfResults['ML_Type'] = clfResults['Clf_type']
clfResults.drop('Clf_type',axis=1, inplace=True)

regResults = setupReg(regResults)
clfResults = setupClf(clfResults)

def plotScorevsXVARLine(Results,x,y,XVars, ML_Type, scoreType,Duplicates_removed,Dates,Algo,Binary, style, hue, size):

    Results = (Results.loc[Results['variable'].isin(scoreType)]
    .loc[Results['ML_Type'].isin(ML_Type)]
    .loc[Results['Duplicates_removed'].isin(Duplicates_removed)]
    .loc[Results['XVars'].isin(XVars)]
    .loc[Results['Algo'].isin(Algo)]
    .loc[Results['Dates'].isin(Dates)]
    )

    if Binary != [None]:
        Results = Results.loc[Results['Binary'].isin(Binary)]


    plt.title(f'{scoreType} of {x} by {y}')
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    sns.set_theme(style='darkgrid')
    sns.lineplot(data=Results, x=x, y=y, style=style,hue = hue, palette='colorblind',ci=None)
    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left', borderaxespad=0)
    plt.show()

def plotScorevsXVARScatter(Results,x,y,XVars, ML_Type, scoreType,Duplicates_removed,Dates,Algo,Binary, style, hue, size):

    Results = (Results.loc[Results['variable'].isin(scoreType)]
        .loc[Results['ML_Type'].isin(ML_Type)]
        .loc[Results['Duplicates_removed'].isin(Duplicates_removed)]
        .loc[Results['XVars'].isin(XVars)]
        .loc[Results['Algo'].isin(Algo)]
        .loc[Results['Dates'].isin(Dates)]
        )

    if Binary != [None]:
        Results = Results.loc[Results['Binary'].isin(Binary)]

    plt.title(f'{scoreType} of {x} by {y}')
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    sns.set_theme(style='darkgrid')
    sns.scatterplot(data=Results, x=x, y=y, style=style, hue=hue, palette='colorblind', ci=None)
    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left', borderaxespad=0)
    plt.show()


def plotScorevsXVARBar(Results, x, y, XVars, ML_Type, scoreType, Duplicates_removed, Dates, Algo, Binary,
                           hue, size):
    Results = (Results.loc[Results['variable'].isin(scoreType)]
        .loc[Results['ML_Type'].isin(ML_Type)]
        .loc[Results['Duplicates_removed'].isin(Duplicates_removed)]
        .loc[Results['XVars'].isin(XVars)]
        .loc[Results['Algo'].isin(Algo)]
        .loc[Results['Dates'].isin(Dates)]
        )

    if Binary != [None]:
        Results = Results.loc[Results['Binary'].isin(Binary)]

    plt.title(f'{scoreType} of {x} by {y}')
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    sns.set_theme(style='darkgrid')
    sns.barplot(data=Results, x=x, y=y, hue=hue, palette='colorblind', ci=None)
    plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left', borderaxespad=0)
    plt.show()

XAuto = ['Auto','AutoMeta','AutoNLP','All','PossibleBest']
XMeta = ['Meta','AutoMeta', 'MetaNLP', 'PossibleBest']
XNLP = ['NLP', 'AutoNLP', 'MetaNLP', 'All', 'PossibleBest']
XBase = ['Auto', 'Meta', 'NLP']
XAutoNLP = ['Auto','AutoNLP']
XMetaNLP = ['Meta','MetaNLP']
XAll = ['Auto','Meta','NLP','AutoMeta','AutoNLP','MetaNLP','All','PossibleBest']

clf_algo = ['clf_GradientBoosting','clf_NN'] #'clf_GradientBoosting','clf_NN','clf_logreg','clf_SGD'
reg_algo = ['reg_GradientBoosting','reg_NN','reg_SGD'] #'reg_GradientBoosting','reg_NN','reg_MLR','reg_SGD'

clf_score = ['Test_acc']
#'Train_acc','Train_prec','Train_recall','Test_acc','Test_prec','Test_recall','Val_acc','Val_prec','Val_recall'
reg_score = []

for i in clfResults.columns: print(i)

#plotScorevsXVARLine(regResults,x='Dates',y='value', scoreType=['Test_MSE','Train_MSE'],XVars=XBase, ML_Type=['CS_Regressor'],Duplicates_removed=[True],
                  #  Algo = reg_algo,Dates=['2010-01-01'], Binary=[None],
               # hue = 'XVars',
               # style='variable',size=None )

#plotScorevsXVARLine(clfResults,x='XVars',y='value',XVars=XBase, ML_Type=['CS_Classifier'],
                    #Duplicates_removed=[True],Algo = clf_algo,Dates=['2010-01-01'],Binary=[True],
                    #scoreType=['Test_acc'],
                #hue = 'Algo',
                #style='Binary',size=None )

plotScorevsXVARScatter(clfResults,x='XVars',y='value',XVars=XAll, ML_Type=['TS_Classifier'],
                    Duplicates_removed=[True],Algo = clf_algo,Dates=['2010-01-01'],Binary=[True],
                    scoreType=clf_score,
                    hue = 'Algo',
                    style='variable',size=None )

plotScorevsXVARBar(clfResults,x='XVars',y='value',XVars=XAll, ML_Type=['TS_Classifier'],
                    Duplicates_removed=[True],Algo = clf_algo,Dates=['2010-01-01'],Binary=[True],
                    scoreType=clf_score,
                    hue = 'Algo'
                    ,size=None )