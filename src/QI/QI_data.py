import os

import pandas as pd
import numpy as np
from numpy.random import choice, seed
from random import random
# from sklearn.preprocessing import StandardScaler

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

# import preprocess
# import preStroked

if __name__ == '__main__':
    # load or construct dataframe.
    if os.name == 'posix':
        csv_file = '../../GN-data/normalStroke.csv'
    else:
        csv_file = '..\\..\\GN-data\\normalStroke.csv'
    if os.path.isfile(csv_file): 
        dft = pd.read_csv(csv_file)
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
        dft.to_csv(csv_file, encoding='utf_8_sig',index=False)
    # spliting the file to train and test sets
    if os.name == 'posix':
        train_fn = '../../GN-data/dnn_train.csv'
        valid_fn = '../../GN-data/dnn_valid.csv'
        test_fn = '../../GN-data/dnn_test.csv'
        trains_fn = '../../GN-data/dnn_train_sf.csv'
        valids_fn = '../../GN-data/dnn_valid_sf.csv'
        tests_fn = '../../GN-data/dnn_test_sf.csv'
        trains_fni = '../../GN-data/dnn_train_sf_ind.csv'
        valids_fni = '../../GN-data/dnn_valid_sf_ind.csv'
        tests_fni = '../../GN-data/dnn_test_sf_ind.csv'
        cate_fn = '../../GN-data/normalCategory.csv'
    else:
        train_fn = '..\\..\\GN-data\\dnn_train.csv'
        valid_fn = '..\\..\\GN-data\\dnn_valid.csv'
        test_fn = '..\\..\\GN-data\\dnn_test.csv'
        trains_fn = '..\\..\\GN-data\\dnn_train_sf.csv'
        valids_fn = '..\\..\\GN-data\\dnn_valid_sf.csv'
        tests_fn = '..\\..\\GN-data\\dnn_test_sf.csv'
        trains_fni = '..\\..\\GN-data\\dnn_train_sf_ind.csv'
        valids_fni = '..\\..\\GN-data\\dnn_valid_sf_ind.csv'
        tests_fni = '..\\..\\GN-data\\dnn_test_sf_ind.csv'
        cate_fn = '..\\..\\GN-data\\normalCategory.csv'
    dft = dft.reset_index(drop=True)
    dft= dft.drop(dft[(dft['档案年度']==2019) & (dft['风险评级']==5)].index)
    dft = dft.drop(dft[(dft['风险评级']==0)].index)
    dft.drop(['身份证号', '档案年度'], axis=1, inplace=True)
    dft['风险评级'] = dft['风险评级'].apply(lambda x: x - 1)
    dft['风险评级'] = dft['风险评级'].apply(lambda x: 3 if x == 4 else x)
    dft = dft.reset_index(drop=True)
    dft = dft.rename(columns={'性别': 'Gd', '建档年龄': 'FA', '民族': 'Nat', '婚姻状况': 'MS', '受教育程度': 'Edu', 
                            '是否退休': 'Ret', '风险评级': 'RR', '卒中': 'Apx', '吸烟': 'Sm', '吸烟年限': 'YS', 
                            '饮酒': 'Drk', '缺乏运动': 'Exs', '口味': 'Flv', '荤素': 'MV', '食用蔬菜': 'VC', 
                            '食用水果': 'FC', '脑卒中': 'HA', '冠心病': 'HCAD', '高血压': 'HEH', '糖尿病': 'HDM',
                            '身高': 'Ht', '体重': 'Wt', '左侧收缩压': 'LSBP', '左侧舒张压': 'LDBP',
                            '右侧收缩压': 'RSBP', '右侧舒张压': 'RDBP', '心律': 'Rhm', '脉搏': 'Pls', 
                            '空腹血糖': 'FBG', '糖化血红蛋白': 'HbA1c', '甘油三脂': 'TG', '总胆固醇': 'TC', 
                            '低密度脂蛋白胆固醇': 'LDL-C', '高密度脂蛋白胆固醇': 'HDL-C', '同型半胱氨酸': 'Hcy'})
   
    
    
    # 实验完毕
    # 切分  
    train_df, valid_df, test_df = data_split(dft, prob=[0.85, 0.8], random_state=20)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(train_fn, encoding='utf_8_sig', index=False)
    valid_df.to_csv(valid_fn, encoding='utf_8_sig', index=False)
    test_df.to_csv(test_fn, encoding='utf_8_sig', index=False)

    # cols = ['LSBP', 'Exs', 'Sm', 'LDBP', 'HbA1c', 'HA', 'FBG', 'TG', 'HDL-C', 'TC', 'LDL-C',
    #          'Edu', 'FA', 'Hcy', 'HEH', 'Ret', 'Flv', 'MV', 
    #          'VC', 'FC', 'HCAD', 'HDM', 'YS', 'Ht', 'Wt', 'BMI',
    #          'RR']

    cols = ['LSBP', 'Exs', 'Sm', 'LDBP', 'RSBP','HbA1c', 'HA', 'FBG', 'TG', 'Wt','HDL-C', 'TC', 'LDL-C',
             'Edu', 'FA', 'Ht', 'RDBP','BMI','Hcy', 'HEH',
             #'Ret', 'Flv', 'MV', 'VC', 'FC', 'HCAD', 'HDM', 'YS',   
             'RR'
             ]
    
    #cols = train_df.columns
    
    train_dfs = train_df.loc[:, cols]
    valid_dfs = valid_df.loc[:, cols]
    test_dfs = test_df.loc[:, cols]
    train_dfs.to_csv(trains_fn, encoding='utf_8_sig', index=False)
    valid_dfs.to_csv(valids_fn, encoding='utf_8_sig', index=False)
    test_dfs.to_csv(tests_fn, encoding='utf_8_sig', index=False)
    
    D = dict()
    feature_sizes = []

    dfidx = dft.copy()

    for col in cols:
        x = dfidx[col].unique()
        feature_sizes.append(len(x))
        # print(col, len(x))
        d = dict()
        for i, v in enumerate(x):
            d[v] = i
        D[col] = d
        dfidx[col] = dfidx[col].apply(lambda x: d[x])

    for col in cols:
        d = D[col]
        train_dfs[col] = train_dfs[col].apply(lambda x: d[x])
        valid_dfs[col] = valid_dfs[col].apply(lambda x: d[x])
        test_dfs[col] = test_dfs[col].apply(lambda x: d[x])

    train_dfs.to_csv(trains_fni, encoding='utf_8_sig', index=False)
    valid_dfs.to_csv(valids_fni, encoding='utf_8_sig', index=False)
    test_dfs.to_csv(tests_fni, encoding='utf_8_sig', index=False)

    n_category = 30
    dftc = dft.copy()
    for i, col in enumerate(cols):
        if col == 'LSBP':
            continue
        if feature_sizes[i] > 30:
            x = dftc[col]
            xmin = x.min()
            xmax = x.max()
            x = x.apply(lambda x: (x-xmin)/(xmax-xmin)*(n_category-1))
            x = x.apply(np.floor).astype(int)
            dftc[col] = x
    dftc.to_csv(cate_fn, encoding='utf_8_sig', index=False)