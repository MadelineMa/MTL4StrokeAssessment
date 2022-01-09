import os
import pandas as pd
import numpy as np
from numpy.random import choice, seed
from random import random

def data_split(sample, prob=[0.85, 0.8], random_state=None):
    """
    Divide the input dataframe to train and test sets randomly such that #train / #test = prob
    sample (dataframe)
    prob (float)
    random_state (None or int)
    """
    # Set random state.
    if random_state is not None:
        seed(random_state)
    # Split data
    n_rows, _ = sample.shape
    k = int(n_rows * prob[0])
    train_valid_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_valid_indexes])
    ktmp = int(k*prob[1])
    tmp_indexes = choice(range(k), size=ktmp, replace=False)
    train_indexes = np.array([train_valid_indexes[i] for i in tmp_indexes])
    valid_indexes = np.array([i for i in train_valid_indexes if i not in train_indexes])
    train_data = sample.loc[list(train_indexes)]
    valid_data = sample.loc[list(valid_indexes)]
    test_data = sample.loc[list(test_indexes)]
    return train_data, valid_data, test_data

if __name__ == '__main__':
    # load or construct dataframe.
    if os.name == 'posix':
        csv_file = '../../GN-data/normal_en.csv'
    else:
        csv_file = '..\\..\\GN-data\\normal_en.csv'
    try:
        dft = pd.read_csv(csv_file)
    except FileExistsError:
        print(csv_file, 'not exists! may run QI_data.py')

    # spliting the file to train and test sets
    if os.name == 'posix':
        train_fn = './data/qi_train.csv'
        valid_fn = './data/qi_valid.csv'
        test_fn = './data/qi_test.csv'
        train_fni = './data/qi_train_ind.csv'
        valid_fni = './data/qi_valid_ind.csv'
        test_fni = './data/qi_test_ind.csv'
    else:
        train_fn = '.\\data\\qi_train.csv'
        valid_fn = '.\\data\\qi_valid.csv'
        test_fn = '.\\data\\qi_test.csv'
        train_fni = '.\\data\\qi_train_ind.csv'
        valid_fni = '.\\data\\qi_valid_ind.csv'
        test_fni = '.\\data\\qi_test_ind.csv'
    dft = dft.reset_index(drop=True)
    
    # cols = ['LSBP', 'Exs', 'Sm', 'LDBP', 'RSBP','HbA1c', 'HA', 'FBG', 'TG', 'Wt','HDL-C', 'TC', 'LDL-C',
    #        'Edu', 'FA', 'Ht', 'RDBP','BMI','Hcy', 'HEH',
    #        'Ret', 'Flv', 'MV', 'VC', 'FC', 'HCAD', 'HDM', 'YS',   
    #        'RR'
    #        ]
    
    # dft = dft.loc[:, cols]
    
    train_df, valid_df, test_df = data_split(dft, prob=[0.85, 0.8], random_state=20)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_fn, encoding='utf_8_sig', index=False)
    valid_df.to_csv(valid_fn, encoding='utf_8_sig', index=False)
    test_df.to_csv(test_fn, encoding='utf_8_sig', index=False)

    feature_sizes = []
    dfidx = dft.copy()

    cols = train_df.columns

    D = dict()
    for col in cols:
        x = dfidx[col].unique()
        feature_sizes.append(len(x))
        d = dict()
        for i, v in enumerate(x):
            d[v] = i
        D[col] = d
        dfidx[col] = dfidx[col].apply(lambda x: d[x])

    for col in cols:
        d = D[col]
        train_df[col] = train_df[col].apply(lambda x: d[x])
        valid_df[col] = valid_df[col].apply(lambda x: d[x])
        test_df[col] = test_df[col].apply(lambda x: d[x])
    cols = cols.tolist()
    feature_sizes.pop(cols.index('RR'))
    cols.pop(cols.index('RR'))
    feature_sizes.pop(cols.index('Apx'))
    np.save('./data/feature_sizes.npy', feature_sizes) 
    train_df.to_csv(train_fni, encoding='utf_8_sig', index=False)
    valid_df.to_csv(valid_fni, encoding='utf_8_sig', index=False)
    test_df.to_csv(test_fni, encoding='utf_8_sig', index=False)
