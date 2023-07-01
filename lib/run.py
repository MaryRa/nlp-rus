import os

import pandas as pd
from code_for_learning.solution import BadUsersBehavior

if __name__ == '__main__':
    path = 'nlp-rus-data/'
    test = pd.read_csv(path + 'test.csv',
                       usecols=['category', 'description'])
    train = pd.read_csv(path + 'train.csv',
                        usecols=['category', 'description', 'is_bad'])

    classifier = BadUsersBehavior(n_jobs=2)
    classifier.fit(train, y_name='is_bad')
    test = classifier.predict_proba(test)

    test = test[['prediction', 'start', 'end']].reset_index()
    test[['index', 'prediction']].to_csv(path + 'prediction.csv',
                                         index=False)
    test[['index', 'start', 'end']].to_csv(path + 'mask_prediction.csv',
                                           index=False)
    tmp = pd.read_csv(path + 'prediction.csv')
    print(tmp.head())
    tmp = pd.read_csv(path + 'mask_prediction.csv')
    print(tmp.head())
    print(os.listdir(path), os.path.abspath(path))
