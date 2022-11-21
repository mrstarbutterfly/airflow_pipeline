import json
import dill
import pandas as pd
import os
from os import listdir
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')

def predict() -> None:
    all_pred = []
    car_id = []

    with open(f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl', 'rb') as file:
        model = dill.load(file)
    
    for f in listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/'+ f) as file:
            d = json.load(file)
            df = pd.DataFrame(d, index = [0]) 
        car_id.append(f.split('.')[0])
        all_pred.append(model.predict(df))

    d = {'car_id' : car_id, 'pred' : [i[0] for i in all_pred]}
    data = pd.DataFrame(d)
    data.set_index('car_id', inplace = True)
    data.to_csv(f'{path}/data/predictions/pred{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
