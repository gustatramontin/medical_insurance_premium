from numpy.core.defchararray import array
from model import PredictValue, ClassifyByExpense

from sys import argv, exit
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

if argv[1] not in ['predict', 'classify']:
    print('First parammeter invalid, must be either predict or classify')
with open(f'{argv[1]}.pickle', "rb") as f:
    model = pickle.load(f)

data = pd.read_csv('input_to_predict.csv')

data['sex'] = data['sex'].replace({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].replace({'yes': 1, 'no': 0})

data_for_dummies = pd.read_csv('template.csv')

data_for_dummies['sex'] = data_for_dummies['sex'].replace({'male': 1, 'female': 0})
data_for_dummies['smoker'] = data_for_dummies['smoker'].replace({'yes': 1, 'no': 0})

data = pd.concat([data_for_dummies, data])
data = pd.get_dummies(data)
data = data[4:]


data_predict = data.drop(['personid'], 1)
array_data = np.array(data_predict)

if argv[1] == 'predict':
    poly = PolynomialFeatures(degree=2)
    array_data = poly.fit_transform(array_data)

output = {
    "personid": [],
    "value": []
}
personids = np.array(data['personid'])
for data, i in zip(array_data, range(len(array_data))):
    value = model.predict([data])
    personid = personids[i]

    output['personid'].append(personid)
    output['value'].append(value[0])

output_pandas = pd.DataFrame.from_dict(output)

output_pandas.to_csv('output_predicted.csv', index=False)

