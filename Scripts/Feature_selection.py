import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from ML_Tests import dataSetup, setupXYvars
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class Shrinkage_Methods:

    def __init__(self, data:pandas.DataFrame, X_variables: list[str],Y_variable: str,  num_features: int=10 ):
        data.dropna(inplace=True)
        self.X_variables = data[X_variables]
        self.Y_variable = data[Y_variable]
        self.num_features = num_features


    def run_ElasticNet(self):

        regr = ElasticNet(alpha=self.alpha,random_state=42, l1_ratio=0.5)
        regr.fit(self.X_variables,self.Y_variable)

        if len(self.X_variables.columns) > 20:
            print(f"Max coefficient: {max(regr.coef_)}\n"
                  f"Average coefficient: {np.mean(regr.coef_)}")

        else:
            for i in range(len(regr.coef_)):
                print(f"{self.X_variables.columns[i]} : {regr.coef_[i]}")

        return regr

    def run_Lasso(self):
        regr = Lasso(alpha=self.alpha,random_state=42)
        regr.fit(self.X_variables,self.Y_variable)

        if len(self.X_variables.columns) > 20:
            print(f"Max coefficient: {max(regr.coef_)}\n"
                  f"Average coefficient: {np.mean(regr.coef_)}")

        else:
            for i in range(len(regr.coef_)):
                print(f"{self.X_variables.columns[i]} : {regr.coef_[i]}")

        return regr

    def run_Ridge(self):
        regr = Ridge(alpha=self.alpha,random_state=42)
        regr.fit(self.X_variables,self.Y_variable)

        if len(self.X_variables.columns) > 20:
            print(f"Max coefficient: {max(regr.coef_[0])}\n"
                  f"Average coefficient: {np.mean(regr.coef_[0])}")
        else:

            for i in range(len(self.X_variables.columns)):
                print(f"{self.X_variables.columns[i]} : {regr.coef_[0][i]}")


        return regr

    def Elastic_Gridsearch(self,l1_ratio,show_coefficients=False,minAlpha=0,maxAlpha=10):
        l1_ratio = l1_ratio
        alphas = range(100*minAlpha,100*maxAlpha,2)
        alphas = [x/100 for x in alphas]
        results_list = []

        for i in alphas:
            regr = ElasticNet(alpha=i, random_state=42, l1_ratio=l1_ratio,fit_intercept=False,selection='random')
            regr.fit(self.X_variables, self.Y_variable)
            results_list.append([np.log(i)] + regr.coef_.tolist())

        colNames = ['alpha']
        for i in self.X_variables.columns:
            colNames.append(i)

        results_df = pd.DataFrame(results_list,columns=colNames)
        colNames.pop(0)
        long_results_df = pd.melt(results_df,id_vars='alpha',value_vars=colNames)

        plt.title(f'L1_ratio:{l1_ratio}')
        plt.xlabel('Log Alpha')
        plt.ylabel('Coefficients')
        sns.set_theme(style='darkgrid')
        sns.lineplot(data = long_results_df,x='alpha', y='value',hue = 'variable', palette='colorblind')

        plt.legend(list(self.X_variables.columns),loc='upper right')
        plt.show()

        if show_coefficients:
            for i in results_df.columns:
                print(f'\n{i}:{results_df.loc[0][i]}')





