from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt 

import pickle
from sys import exit

class DataVisualisation:
    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv('insurance.csv')

        data['sex'] = data['sex'].replace({'male': 1, 'female': 0})

        data['smoker'] = data['smoker'].replace({'yes': 1, 'no': 0})

        data = pd.get_dummies(data)

        return data
    def image(self):
        f, ax = plt.subplots(figsize=(9, 6))
        return ax
    def heatmap(self):
        ax = self.image()
        sns.heatmap(self.data.corr(), annot=True, linewidths=.5, ax=ax)
        plt.show()

    def pairplot(self):
        ax = self.image()
        sns.pairplot(self.data)
        plt.show()

    def scatter(self, column1, column2):

        mean_data = self.data.groupby(column1, as_index=False)[column2].mean()


        plt.scatter(mean_data[column1], mean_data[column2])

        plt.xlabel(column1)
        plt.ylabel(column2)

        plt.show()

    def bar(self,column1, column2):

        mean_data = self.data.groupby(column1, as_index=False)[column2].mean()

        plt.bar(mean_data[column1], mean_data[column2])

        plt.xlabel(column1)
        plt.ylabel(column2)

        plt.show()


class PredictValue:
    def __init__(self):
        self.model = None

    def load_data(self):

        data = pd.read_csv('insurance.csv')

        data['sex'] = data['sex'].replace({'male': 1, 'female': 0})

        data['smoker'] = data['smoker'].replace({'yes': 1, 'no': 0})

        data = pd.get_dummies(data)

        x = np.array(data.drop(['expenses'], 1))
        y = np.array(data['expenses'])

        return (x, y)

    def generate_model(self):
        x, y = self.load_data()
        best = 0
        model = None
        for _ in range(100):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

            linear = LinearRegression()
            linear.fit(x_train, y_train)
            
            acc = linear.score(x_test, y_test)

            if acc > best:
                best = acc
                model = linear

        return model
    def generate_model_poly(self):
        x, y = self.load_data()

        best = 0
        model = None
        poly = PolynomialFeatures(degree=2)
        for _ in range(100):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

            x_train =  poly.fit_transform(x_train)
            x_test = poly.fit_transform(x_test)
            linear = LinearRegression()
            linear.fit(x_train, y_train)
            
            acc = linear.score(x_test, y_test)

            if acc > best:
                best = acc
                model = linear

        self.model = model
        return model

    def predict(self, x, model=None):
        if model == None and self.model != None:
            model = self.model
        else:
            raise ValueError('You need either pass model as parameter or generate one using the methods!')

        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)
        y_pred = model.predict(x_poly)

        return y_pred

    def mean_absolute_percentage_error(self):

        x, y_true = self.load_data()

        y_pred = self.predict(x)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def save(self, name):
        with open(f"{name}.pickle", "wb") as f:
            pickle.dump(self.model, f)

class ClassifyByExpense:
    def __init__(self, expenseToPremium):
        self.expenseToPremium = expenseToPremium # Expense value in which premium is recomended

    def load_data(self):

        data = pd.read_csv('insurance.csv')

        data['sex'] = data['sex'].replace({'male': 1, 'female': 0})

        data['smoker'] = data['smoker'].replace({'yes': 1, 'no': 0})

        data['expenses'].values[data['expenses'] < self.expenseToPremium] = 0
        data['expenses'].values[data['expenses'] >= self.expenseToPremium] = 1

        data = pd.get_dummies(data)

        x = np.array(data.drop(['expenses'], 1))
        y = np.array(data['expenses'])

        return (x, y)

    def generate_model(self):
        x, y = self.load_data()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

        forest = RandomForestClassifier(n_estimators=23)
        forest.fit(x_train, y_train)
            
        if __name__ == "__main__":
            acc = forest.score(x_test, y_test)
            print(f'Acurracy RandomForest: {acc*100}')

        self.model = forest
        return forest

    def predict(x):
        y_pred = self.model.predict(x)

        return y_pred

    def save(self, name):
        with open(f"{name}.pickle", "wb") as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    with open('premium_expense.txt', 'r') as f:
        premium_expense = f.read()
        try:
            premium_expense = int(premium_expense)
        except:
            print('premium_expense.txt: it\'s not a number')
            exit()

    
    classify = ClassifyByExpense(premium_expense)
    classify.generate_model()
    
    regression = PredictValue()
    regression.generate_model_poly()
    print(f'Mean difference between real and predicted: {regression.mean_absolute_percentage_error()}')

    classify.save('classifier')
    regression.save('predictor')
    print('Saved')

    #analyses = DataVisualisation()

    #analyses.heatmap()
    #analyses.pairplot()
    
    #analyses.scatter('age', 'expenses')
    #analyses.bar('smoker','expenses')
    
    #analyses.scatter('bmi', 'expenses')
    #analyses.bar('children', 'expenses')

