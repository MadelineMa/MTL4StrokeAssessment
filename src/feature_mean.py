import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    # load or construct dataframe.
    if os.name == 'posix':
        csv_file = '../../GN-data/normal_en.csv'
    else:
        csv_file = '..\\..\\GN-data\\normal_en.csv'
    if os.path.isfile(csv_file): 
        df = pd.read_csv(csv_file)
    else:
        FileExistsError(csv_file+'not exist!')

    # cols = ['LSBP', 'LDBP', 'Exs', 'Sm', 'HA', 'RR']

    # cols = ['FBG', 'HbA1c', 'TC', 'TG', 'Hcy', 'HDL-C', 'LDL-C', 'Wt', 'BMI', 'MV', 'RR'] 
    cols = ['FBG', 'HbA1c', 'Hcy', 'MV']

    df = df[cols]

    df.groupby("RR").agg('mean')

    # hard code for plots
    plt.subplot(1,4,1)
    # Sm [0.328071, 0.193252, 0.784658, 0.817489] -> [0.193252, 0.134819, 0.456586, 0.032831]
    plt.bar(1, 0.193252, label='Medium', color='m', width=0.2)
    plt.bar(1, 0.134819, bottom=0.193252, label='Low', color='r', width=0.2)
    plt.bar(1, 0.456586, bottom=0.328071, label='High', color='b', width=0.2)
    plt.bar(1, 0.032831, bottom=0.784658, label='Stroke', color='g', width=0.2)
    plt.plot([0.85, 1.15], [0.0, 0.0], '--', color='tomato', linewidth=2.0)
    plt.yticks([0, 0.19, 0.33, 0.78, 0.81], labels=['0', '0.19', '0.33', '0.78', '0.81'])
    plt.xticks([1], labels=['Sm'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.get_yticklabels()[0].set_color("tomato")
    # plt.spines['right'].set_visible(False)
    # plt.spines['top'].set_visible(False)

    # plt.legend(loc='upper center')

    plt.subplot(1,4,2)
    # Rhm [0.010940, 0.022154, 0.023836, 0.035587] -> [0.010940, 0.011214, 0.001682, 0.011751]
    plt.bar(1, 0.010940, label='Low', color='r', width=0.2)
    plt.bar(1, 0.011214, bottom=0.010949, label='Medium', color='m', width=0.2)
    plt.bar(1, 0.001682, bottom=0.022154, label='High', color='b', width=0.2)
    plt.bar(1, 0.011751, bottom=0.023836, label='Stroke', color='g', width=0.2)
    plt.plot([0.85, 1.15], [0.0, 0.0], '--', color='tomato', linewidth=2.0)
    plt.yticks([0, 0.01, 0.022, 0.036], labels=['0', '0.01', '0.022', '0.036'])
    plt.xticks([1], labels=['Rhm'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.get_yticklabels()[0].set_color("tomato")

    plt.subplot(1,4,3)
    # FBG [4.629298, 5.075433, 5.404412, 6.230831] -> [0.446135, 0.328979, 0.826419]
    plt.bar(1, 0.629298, label='Low', color='r', width=0.2)
    plt.bar(1, 0.446135, bottom=0.629298, label='Medium', color='m', width=0.2)
    plt.bar(1, 0.328979, bottom=1.075433, label='High', color='b', width=0.2)
    plt.bar(1, 0.826419, bottom=1.404412, label='Stroke', color='g', width=0.2)
    plt.plot([0.85, 1.15], [2.1, 2.1], '--', color='tomato', linewidth=2.0, label='Health')
    plt.yticks([0.63, 1.08, 1.40, 2.1, 2.23], labels=['4.63', '5.08', '5.40', '6.1', '6.23'])
    plt.xticks([1], labels=['FBG'])
    # plt.legend(loc='lower center')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.get_yticklabels()[3].set_color("tomato")

    plt.subplot(1,4,4)
    # HbA1c [5.301732, 5.525905, 5.804787, 6.582822] -> [0.224173, 0.278882, 0.778035]
    plt.bar(1, 0.301732, label='Low', color='r', width=0.2)
    plt.bar(1, 0.224173, bottom=0.301732, label='Medium', color='m', width=0.2)
    plt.bar(1, 0.278882, bottom=0.525905, label='High', color='b', width=0.2)
    plt.bar(1, 0.778035, bottom=0.804787, label='Stroke', color='g', width=0.2)
    plt.plot([0.85, 1.15], [1.0, 1.0], '--', color='tomato', linewidth=2.0, label='Health')
    plt.yticks([0.30, 0.53, 0.80, 1, 1.58], labels=['5.30', '5.53', '5.80', '6', '6.58'])
    plt.xticks([1], labels=['HbA1c'])
    plt.legend(loc='upper center')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.get_yticklabels()[3].set_color("tomato")

    # plt.subplot(1,5,5)
    # # Drk [0.144301, 0.113667, 0.230913, 0.458058] -> [0.030634, 0.117246, 0.227145]
    # plt.bar(1, 0.113667, label='Medium', color='m', width=0.2)
    # plt.bar(1, 0.030634, bottom=0.113667, label='Low', color='r', width=0.2)
    # plt.bar(1, 0.117246, bottom=0.144301, label='High', color='b', width=0.2)
    # plt.bar(1, 0.227145, bottom=0.230913, label='Stroke', color='g', width=0.2)
    # plt.plot([0.85, 1.15], [0.0, 0.0], '--', color='tomato', linewidth=2.0, label='Health')
    # plt.yticks([0, 0.12, 0.23, 0.46], labels=['0', '0.12', '0.23', '0.46'])
    # plt.xticks([1], labels=['Drk'])
    # plt.legend(loc='upper center')
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.get_yticklabels()[0].set_color("tomato")

    # TODO: zoom in to 'FBG' and 'HbA1c' to try the motivation of stroke state, 
    #       and the motivation may mainly contain the transition state prediction
    #       The stroke state is suitable to be an individual state along with low/medium/high-level risks evidenced by the features. 
    # TODO: bar plot for the above problem: this may require normalize the values.

    cols = ['Drk', 'RR']
    # dfs = df[['Rhm', 'RR']]
    dfs = df[cols]
    # dfs.value_counts()
    dfs = dfs.drop(dfs[dfs[cols[0]]==-1].index)
    dfs.groupby("RR").agg('mean')