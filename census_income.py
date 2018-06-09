import tensorflow as tf
import pandas as pd
import pprint

data_path = ('./datasets/census_income_dataset.data')

headers = ['AGE', 'WORKCLASS', 'FNLWGT', 'EDUCATION', 'EDUCATION NUM',
           'MARITAL STATUS', 'OCCUPATION', 'RELATIONSHIP', 'RACE', 'SEX',
           'CAPITAL GAIN', 'CAPITAL LOSS', 'HOURS PER WEEK', 'NATIVE COUNTRY']

census_data = pd.read_csv(data_path, header=None, names=headers)
pprint.pprint(census_data)
