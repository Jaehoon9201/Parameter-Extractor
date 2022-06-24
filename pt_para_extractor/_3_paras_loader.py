

from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
from torch import cuda
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import seaborn as sns
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from _0_utilities import FaultClassifier
from _0_utilities import Train_Dataset, Test_Dataset
from _0_utilities import plot_func1, plot_func2
from _0_utilities import hid1_num, hid2_num, out_num
import pandas as pd
import os


os.makedirs('./best_paras', exist_ok=True)

eval_all_file = pd.read_csv("test_results/evals_all.csv", index_col=0)
# BEST_DATA_CASE = eval_all_file.loc['BEST ITER MODEL INDEX'][0]
BEST_CASE = eval_all_file.loc['BEST ACC MEAN ALL'][0]
# getting the best iter index
data_case_eval_all_file = pd.read_csv("test_results/data_case_%s_evals_all.csv" %(BEST_CASE), index_col=0)
BEST_ITER = data_case_eval_all_file.loc['BEST ITER MODEL INDEX'][0]

# PRINTING
print('==================================')
print('     BEST CASE DATA  :  ',  BEST_CASE)
print('     BEST TER       :  ',  BEST_ITER)
print('==================================\n')

# laod the best model using the above index
falut_classifier = FaultClassifier()
falut_classifier = torch.load('train_results/datacase_%s_iter_%s_model.pt' %(BEST_CASE, BEST_ITER))

l1_weights = falut_classifier.l1.weight.data.cpu().numpy()
l2_weights = falut_classifier.l2.weight.data.cpu().numpy()
l3_weights = falut_classifier.l3.weight.data.cpu().numpy()

l1_bias = falut_classifier.l1.bias.data.cpu().numpy()
l2_bias = falut_classifier.l2.bias.data.cpu().numpy()
l3_bias = falut_classifier.l3.bias.data.cpu().numpy()


for i in range(0, hid1_num):
    globals()['l1_weights_{}'.format(i)] = l1_weights[i].tolist()

for i in range(0, hid2_num):
    globals()['l2_weights_{}'.format(i)] = l2_weights[i].tolist()

for i in range(0, out_num):
    globals()['l3_weights_{}'.format(i)] = l3_weights[i].tolist()



# ================================================
# =================== weights ====================
# ================================================
cnt= 0
with open("best_paras/l1_weights.txt", 'w') as output:
    for i in range(0, hid1_num):
        output.write('{')
        for row in globals()['l1_weights_{}'.format(i)]:
            cnt += 1
            if cnt != hid1_num:
                output.write('  '+str(np.round(row, 7)) +',' )
            else :
                cnt = 0
                output.write('  ' + str(np.round(row, 7)) )
        output.write('}'+ "\n")


with open("best_paras/l2_weights.txt", 'w') as output:
    for i in range(0, hid2_num):
        output.write('{')
        for row in globals()['l2_weights_{}'.format(i)]:
            cnt += 1
            if cnt != hid2_num:
                output.write('  ' + str(np.round(row, 7)) + ',')
            else:
                cnt = 0
                output.write('  ' + str(np.round(row, 7)))
        output.write('}'+ "\n")

with open("best_paras/l3_weights.txt", 'w') as output:
    for i in range(0, out_num):
        output.write('{')
        for row in globals()['l3_weights_{}'.format(i)]:
            cnt += 1
            if cnt != out_num:
                output.write('  ' + str(np.round(row, 7)) + ',')
            else:
                cnt = 0
                output.write('  ' + str(np.round(row, 7)))
        output.write('}' + "\n")

# ================================================
# ===================== bias =====================
# ================================================

with open("best_paras/l1_bias.txt", 'w') as output:
    output.write('{')
    for row in l1_bias:
        cnt += 1
        if cnt != hid1_num:
            output.write('  ' + str(np.round(row, 7)) + ',')
        else:
            cnt = 0
            output.write('  ' + str(np.round(row, 7)))
    output.write('}'+ "\n")

with open("best_paras/l2_bias.txt", 'w') as output:
    output.write('{')
    for row in l2_bias:
        cnt += 1
        if cnt != hid2_num:
            output.write('  ' + str(np.round(row, 7)) + ',')
        else:
            cnt = 0
            output.write('  ' + str(np.round(row, 7)))
    output.write('}'+ "\n")

with open("best_paras/l3_bias.txt", 'w') as output:
    output.write('{')
    for row in l3_bias:
        cnt += 1
        if cnt != out_num:
            output.write('  ' + str(np.round(row, 7)) + ',')
        else:
            cnt = 0
            output.write('  ' + str(np.round(row, 7)))
    output.write('}' + "\n")


# ================================================
# ================== C-script ====================
# ================================================



# =================== weights ====================
cnt= 0
with open("best_paras/c_script.txt", 'w') as output:
    output.write('float ann_theta1_bT[8][8] =  { ' + "\n")
    for i in range(0, hid1_num):
        output.write('{')
        for row in globals()['l1_weights_{}'.format(i)]:
            cnt += 1
            if cnt != hid1_num:
                output.write('  '+str(np.round(row, 7)) +',' )
            else :
                cnt = 0
                output.write('  ' + str(np.round(row, 7)) )
        if i == hid1_num-1 :
            output.write('}'+ "\n")
        else :
            output.write('},' + "\n")
    output.write('}'+ "\n"+ "\n"+ "\n")


with open("best_paras/c_script.txt", 'a') as output:
    output.write('float ann_theta2_bT[8][8] =  { ' + "\n")
    for i in range(0, hid2_num):
        output.write('{')
        for row in globals()['l2_weights_{}'.format(i)]:
            cnt += 1
            if cnt != hid2_num:
                output.write('  ' + str(np.round(row, 7)) + ',')
            else:
                cnt = 0
                output.write('  ' + str(np.round(row, 7)))
        if i == hid2_num-1:
            output.write('}' + "\n")
        else:
            output.write('},' + "\n")
    output.write('}' + "\n" + "\n" + "\n")

with open("best_paras/c_script.txt", 'a') as output:
    output.write('float ann_theta3_bT[3][8] =  { ' + "\n")
    for i in range(0, out_num):
        output.write('{')
        for row in globals()['l3_weights_{}'.format(i)]:
            cnt += 1
            if cnt != out_num:
                output.write('  ' + str(np.round(row, 7)) + ',')
            else:
                cnt = 0
                output.write('  ' + str(np.round(row, 7)))
        if i == out_num-1:
            output.write('}' + "\n")
        else:
            output.write('},' + "\n")
    output.write('}' + "\n" + "\n" + "\n")


# ===================== bias =====================

with open("best_paras/c_script.txt", 'a') as output:
    output.write('float ann_bias1[8] =  ' )
    output.write('{')
    for row in l1_bias:
        cnt += 1
        if cnt != hid1_num:
            output.write('  ' + str(np.round(row, 7)) + ',')
        else:
            cnt = 0
            output.write('  ' + str(np.round(row, 7)))
    output.write('}'+ "\n"+ "\n"+ "\n")

with open("best_paras/c_script.txt", 'a') as output:
    output.write('float ann_bias2[8] =  ')
    output.write('{')
    for row in l2_bias:
        cnt += 1
        if cnt != hid2_num:
            output.write('  ' + str(np.round(row, 7)) + ',')
        else:
            cnt = 0
            output.write('  ' + str(np.round(row, 7)))
    output.write('}'+ "\n"+ "\n"+ "\n")

with open("best_paras/c_script.txt", 'a') as output:
    output.write('float ann_bias3[3] =  ')
    output.write('{')
    for row in l3_bias:
        cnt += 1
        if cnt != out_num:
            output.write('  ' + str(np.round(row, 7)) + ',')
        else:
            cnt = 0
            output.write('  ' + str(np.round(row, 7)))
    output.write('}' + "\n" + "\n" + "\n")
