import os
import sys
import numpy as np
import prettytable as pt
if os.name == 'posix':
    sys.path.append("../")
else:
    sys.path.append("..\\")
from utils import statistics

# tb = pt.PrettyTable()

def res_table(fn, model_type, num_feature):
    try:
        with np.load(fn) as f:
            PScr, RScr, FScr, nIter = f['PScr'], f['RScr'], f['FScr'], f['nIter']
            # PScr, RScr, FScr, nIter = f['PScr2'], f['RScr2'], f['FScr2'], f['nIter']
    except FileNotFoundError:
        print('Error: {} not found!'.format(fn))
    # PScr = PScr.mean(axis=1)

    # print('max(R)={}'.format(max(RScr[:,-1])))
    mPR, sPR, PRCIL, PRCIR = statistics(RScr[:,-1])
    tb = pt.PrettyTable()
    tb.field_names = [model_type, "R(s)", "std", "CIL", "CIR"]
    tb.add_row([num_feature, mPR, sPR, PRCIL, PRCIR])
    print(tb)
    PScr, RScr, FScr = PScr.mean(axis=1), RScr.mean(axis=1), FScr.mean(axis=1)
    mP, sP, PCIL, PCIR = statistics(PScr)
    mR, sR, RCIL, RCIR = statistics(RScr)
    mF, sF, FCIL, FCIR = statistics(FScr)
    mI, sI, ICIL, ICIR = statistics(nIter)
    tb = pt.PrettyTable()
    tb.field_names = [model_type, "precision", "std", "CIL", "CIR"]
    tb.add_row([num_feature, mP, sP, PCIL, PCIR])
    print(tb)

    tb = pt.PrettyTable()
    tb.field_names = [model_type, "Recall", "std", "CIL", "CIR"]
    tb.add_row([num_feature, mR, sR, RCIL, RCIR])
    print(tb)

    tb = pt.PrettyTable()
    tb.field_names = [model_type, "F1-score", "std", "CIL", "CIR"]
    tb.add_row([num_feature, mF, sF, FCIL, FCIR])
    print(tb)

    tb = pt.PrettyTable()
    tb.field_names = [model_type, "#iteration", "std", "CIL", "CIR"]
    tb.add_row([num_feature, mI, sI, ICIL, ICIR])
    print(tb)


    # tb = pt.PrettyTable()
    # tb.field_names = [model_type, "mean(f1)", "mean(P)", "mean(R)", "max(R)", "mean(Ra)", "#Iter"]
    # tb.add_row([num_feature, FScr.mean(axis=0).mean(), PScr.mean(axis=0).mean(), RScr.mean(axis=0).mean(), max(RScr[:,-1]), RScr[:,-1].mean(), nIter.mean()])
    # print(tb)

fn = "./dnn/npz/dnn_risk.npz"
res_table(fn, 'dnn', '34')
fn = "./QI/npz/qi_cat_3.npz"
res_table(fn, 'QI', '34+3')
fn = "./QI/npz/qi_cat_4.npz"
res_table(fn, 'QI', '34+4')
fn = "./QI/npz/qi_cat_5.npz"
res_table(fn, 'QI', '34+5')
fn = "./QI/npz/qi_cat_6.npz"
res_table(fn, 'QI', '34+6')
fn = "./QI/npz/qi_cat_7.npz"
res_table(fn, 'QI', '34+7')
fn = "./QI/npz/qi_cat_20.npz"
res_table(fn, 'QI', '20+3')