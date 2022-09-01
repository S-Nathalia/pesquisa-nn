from functions import calculate_train_size as cal
import pandas as pd

dia = pd.read_csv('../../data/diabetes.csv')
data2 = pd.read_csv('../../data/heart.csv')
data3 = data = pd.read_csv('../../data/water_potability.csv')
qnt_data = 0

while qnt_data < 4000:
    qnt_data += 300
    print(cal(dia, qnt_data))

