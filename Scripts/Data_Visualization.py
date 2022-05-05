import pandas as pd
from matplotlib import pyplot as plt
import time
plt.style.use('seaborn')
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from ML_Tests import Y_cat_format,dataSetup

MLR = LinearRegression(n_jobs=-1)

full_df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(106774, 237).csv')
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values(by='Date', inplace=True)

def runMLR(XVars,YVar,data):

    X = data[XVars]
    Y = data[YVar]

    XY = pd.DataFrame(X, columns=XVars)
    XY['Y'] = pd.DataFrame(Y)
    XY.dropna(inplace=True)
    X = np.array(XY[XVars])
    Y = np.array(XY['Y'])

    MLR.fit(X,Y)

    print(MLR.score(X,Y))

Vs=[]
for i in full_df.columns:
    #print(i)
    if 'V_' in i: Vs.append(i)

print(full_df['VaderPos'].mean())
print(f'Mean of VaderPos: {full_df["VaderPos"].mean()} \n'
      f'Mean of VaderNeg: {full_df["VaderNeg"].mean()} \n'
      f'Mean of VaderNeu: {full_df["VaderNeu"].mean()} \n'
      f'Mean of VaderComp: {full_df["VaderComp"].mean()} \n'
      f'Dif means of VaderPos and VaderNeg: {full_df["VaderPos"].mean() - full_df["VaderNeg"].mean()}\n\n')

print(f'Mean of blobPol: {full_df["blobPol"].mean()} \n'
      f'Mean of blobSubj: {full_df["blobSubj"].mean()}')

xVars = ['DlogDif_1','DlogDif_2','blackSwan_SD3_1','blackSwan_SD4_1','blackSwan_SD5_1','stdVol_1DateResid']
xAutoVars = []
xNLPVars = ['VaderNeu','VaderComp','VaderPos','VaderNeg','blobPol','blobSubj','logDif_date_resid']
xMetaVars = []
yVars = ['logDif_date_resid']

#runMLR(Vs,yVars[0],full_df)

for x in xNLPVars:
    break
    plt.hist(full_df[x])
    plt.title(f'Histogram for {x}')
    plt.xlabel(i)
    plt.show()
    for y in yVars:
        t1 = time.time()
        full_df.sort_values(by=x, inplace=True)
        print(f'X = {x} Y = {y}')
        plt.scatter(full_df[x],full_df[y])
        plt.title(f'{x} vs {y}')
        plt.xlabel(x)
        plt.xticks(rotation=90)
        plt.ylabel(y)
        plt.rcParams["figure.figsize"] = (20, 20)
        plt.show()

#plt.plot(full_df['Date'],full_df['VaderPos'])
#plt.show()
#plt.plot(full_df['Date'],full_df['VaderPos'])
#plt.plot(full_df['Date'],full_df['VaderPos'])
#plt.plot(full_df['Date'],full_df['VaderPos'])
#plt.plot(full_df['Date'],full_df['blobPol'])
#plt.plot(full_df['Date'],full_df['blobSubj'])


df = pd.read_csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset(106774, 237).csv')

startDate = '2010-01-01'
YVar = 'logDif_date_resid'


train,test,val = dataSetup(df,startDate)

all = train.append(test).append(val)

Y_train = Y_cat_format(train,YVar)
Y_test = Y_cat_format(test,YVar)
Y_val = Y_cat_format(val,YVar)
all['YVar'] =  Y_cat_format(all, YVar)
all.sort_values(by= 'Date',inplace=True)

plot1 = plt
plot1.title(f'Train Histogram')
plot1.hist(Y_train, bins = 8)
plot1.show()

plot2 = plt
plot2.title(f'Test Histogram')
plot2.hist(Y_test, bins = 8)
plot2.show()

plot3 = plt
plot3.title(f'Val Histogram')
plot3.hist(Y_val, bins = 8)
plot3.show()

plot4 =plt
plot4.title(f'All Histogram')
plot4.hist(all['YVar'], bins = 8)
plot4.show()

plot5 = plt
plot5.scatter(all['Date'],all['YVar'])
plot5.title('Date vs YVar')
plot5.show()

for i in (all['YVar'],Y_train,Y_test,Y_val):
    print(f'Mean:{np.mean(i)}, Variance:{np.var(i)}\n\n')

