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
    if os.path.isfile(csv_file): 
        dft = pd.read_csv(csv_file)
    else:
        FileExistsError(csv_file+'not exist!')

    if os.name == 'posix':
        train_fn = './data/train_en.csv'
        valid_fn = './data/valid_en.csv'
        test_fn = './data/test_en.csv'
        trains_fni = './data/train_en_ind.csv'
        valids_fni = './data/valid_en_ind.csv'
        tests_fni = './data/test_en_ind.csv'
    else:
        train_fn = '.\\data\\train_en.csv'
        valid_fn = '.\\data\\valid_en.csv'
        test_fn = '.\\data\\test_en.csv'
        trains_fni = '.\\data\\train_en_ind.csv'
        valids_fni = '.\\data\\valid_en_ind.csv'
        tests_fni = '.\\data\\test_en_ind.csv'
    train_df, valid_df, test_df = data_split(dft, prob=[0.85, 0.8], random_state=20)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    cols = ['LSBP', 'Exs', 'Sm', 'LDBP', 'RSBP','HbA1c', 'HA', 'FBG', 'TG', 'Wt','HDL-C', 'TC', 'LDL-C',
             'Edu', 'FA', 'Ht', 'RDBP','BMI','Hcy', 'HEH',
             #'Ret', 'Flv', 'MV', 'VC', 'FC', 'HCAD', 'HDM', 'YS',   
             'RR', 'Apx'
             ]

    dft = dft.loc[:, cols]
    
    train_df = train_df.loc[:, cols]
    valid_df = valid_df.loc[:, cols]
    test_df = test_df.loc[:, cols]

    train_df.to_csv(train_fn, encoding='utf_8_sig', index=False)
    valid_df.to_csv(valid_fn, encoding='utf_8_sig', index=False)
    test_df.to_csv(test_fn, encoding='utf_8_sig', index=False)

    # cols = train_df.columns

    D = dict()
    feature_sizes = []

    dfidx = dft.copy()

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

    train_df.to_csv(trains_fni, encoding='utf_8_sig', index=False)
    valid_df.to_csv(valids_fni, encoding='utf_8_sig', index=False)
    test_df.to_csv(tests_fni, encoding='utf_8_sig', index=False)

    n_category = 30
    dftc = dft.copy()
    for i, col in enumerate(cols):
        if feature_sizes[i] > 30:
            x = dftc[col]
            xmin = x.min()
            xmax = x.max()
            x = x.apply(lambda x: (x-xmin)/(xmax-xmin)*(n_category-1))
            x = x.apply(np.floor).astype(int)
            dftc[col] = x
    dftc.to_csv('./Data/normalCategory.csv', encoding='utf_8_sig', index=False)
    np.save('./data/feature_sizes.npy', feature_sizes)
    # cols = cols.tolist()
    feature_sizes.pop(cols.index('RR'))
    cols.remove('RR')
    feature_sizes.pop(cols.index('Apx'))
