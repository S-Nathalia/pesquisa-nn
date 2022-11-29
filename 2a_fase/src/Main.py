from functions import calculate_train_size as cal
from sklearn.impute import SimpleImputer
from functions import write_file
from sklearn import preprocessing
from Experiment import *
import tensorflow as tf
import pandas as pd
from Model import *

#cancer
cancer = pd.read_csv('../../data/cancer_att.csv')
cancer_path = '../results/experiments - cancer '

#credit card
custumer = pd.read_csv('../../data/customer_data.csv')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(custumer[['fea_2']])
custumer[['fea_2']] = (imp.transform(custumer[['fea_2']]))
custumer_path = '../results/experiments - credit card '

#heart desease
le = preprocessing.LabelEncoder()
heart = pd.read_csv('../../data/heart.csv')
cols = list(heart.select_dtypes(include='object'))
for column in cols:
    le.fit(heart[column])
    heart[column] = le.transform(heart[column])
cols2 = list(heart.select_dtypes(include='float64'))
heart_path = '../results/experiments - heart '

#indian diabetes
indian = pd.read_csv('../../data/diabetes.csv')
indian_path = '../results/experiments - indian_diabets '
