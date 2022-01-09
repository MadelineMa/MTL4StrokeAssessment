import os

import pandas as pd
import numpy as np
from numpy.random import choice, seed
from random import random
# from sklearn.preprocessing import StandardScaler

def train_test_split(sample, prob=0.7, random_state=None):
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
    k = int(n_rows * prob)
    train_indexes = choice(range(n_rows), size=k, replace=False)
    test_indexes = np.array([i for i in range(n_rows) if i not in train_indexes])
    train_data = sample.loc[list(train_indexes)]
    test_data = sample.loc[list(test_indexes)]
    # test_x, test_y = test_data, test_data.pop('风险评级') 
    # train_x, train_y = train_data, train_data.pop('风险评级')
    # return train_x, train_y, test_x, test_y 
    return train_data, test_data

# import preprocess
# import preStroked

if __name__ == '__main__':
    # load or construct dataframe.
    if os.name == 'posix':
        csv_file = '../GN-data/normalStroke.csv'
    else:
        csv_file = '..\\GN-data\\normalStroke.csv'
    if os.path.isfile(csv_file): 
        dft = pd.read_csv(csv_file, index_col=[0])
    else:
        import preprocess
        import preStroked
        dfs = preStroked.dfs
        df = preprocess.df

        df.drop('档案号', axis=1, inplace=True) 
        df = df.rename(columns={'第一次收缩压SBP': '左侧收缩压', '第一次舒张压DBP': '左侧舒张压', '第二次收缩压SBP': '右侧收缩压', '第二次舒张压DBP': '右侧舒张压'})
        col_name=df.columns.tolist()
        col_name.insert(31,'脉搏')
        df=df.reindex(columns=col_name)
        df['脉搏'] = df[['第一次脉搏', '第二次脉搏']].apply(np.mean, axis=1)
        df['脉搏'] = df['脉搏'].apply(np.ceil).astype(int) 
        df.drop(['第一次脉搏', '第二次脉搏'], axis=1, inplace=True) 

        dft = pd.concat([df, dfs], sort=False)

        dft = dft.drop(dft[dft['建档年龄']>120].index)
        dft = dft.drop(dft[(dft['脉搏']<40) | (dft['脉搏']>130)].index)

        dft = dft.drop(dft[(dft['左侧收缩压'] < dft['左侧舒张压']) | (dft['右侧收缩压'] < dft['右侧舒张压'])].index)
        dft.to_csv(csv_file, encoding='utf_8_sig')
    # spliting the file to train and test sets
    if os.name == 'posix':
        train_fn = '../GN-data/stroke_train.csv'
        test_fn = '../GN-data/stroke_test.csv'
    else:
        train_fn = '..\\GN-data\\stroke_train.csv'
        test_fn = '..\\GN-data\\stroke_test.csv'
    # dft.drop(['身份证号', '档案年度'], axis=1, inplace=True)
    # To remove dft['风险评级']==0 & remove dft['档案年度']==2019
    dft = dft.reset_index(drop=True)
    dft= dft.drop(dft[dft['风险评级']==0].index)
    dft= dft.drop(dft[(dft['档案年度']==2019) & (dft['风险评级']==5)].index)
    dft.drop(['身份证号', '档案年度'], axis=1, inplace=True)
    dft['风险评级'] = dft['风险评级'].apply(lambda x: 1 if x==5 else 0)
    dft = dft.reset_index(drop=True)
    # 切分  
    # train_x, train_y, test_x, test_y = train_test_split(dft, prob=0.8, random_state=20)
    train_df, test_df = train_test_split(dft, prob=0.8, random_state=20)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # train_df = train_df.reindex()
    # test_df = test_df.reindex()
    train_df.to_csv(train_fn, encoding='utf_8_sig')
    test_df.to_csv(test_fn, encoding='utf_8_sig')
    # 归一化
    # scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # test_x = scaler.transform(test_x)