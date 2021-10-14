'''
@Author:Dong Yi
@Date:2020.7.8
@Description: 
'''
import math
import random
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import cross_val_score
from MakeSimilarityMatrix_modify import MakeSimilarityMatrix
import sortscore
import tensorflow as tf


# 定义计算circRNA前二十个相似circRNA的方法
def top_fif_sim_circ(circ_sim_matrix, u):

    top_circ_list = np.argsort(-circ_sim_matrix[u,:])
    top_fif_circ_list = top_circ_list[:20]
    top_fif_circ_list = top_fif_circ_list.tolist()
    if u in top_fif_circ_list:
        top_fif_circ_list = top_circ_list[:21]
        top_fif_circ_list = top_fif_circ_list.tolist()
        top_fif_circ_list.remove(u)

    return top_fif_circ_list

def find_associate_disease(top_fif_circ_list, relmatrix, associate_disease_set):
    for i in range(20):
        circ_id = top_fif_circ_list[i]
        for j in range(89):
            if relmatrix[circ_id,j]==1:
                associate_disease_set.add(j)

    return associate_disease_set

# def compute_circ_sim(circ_gipsim_matrix):
#     with h5py.File('../Data/circ_file/circRNA_expression_sim.h5', 'r') as hf:
#         circ_exp_sim = hf['infor'][:]
#     circ_sim_matrix = np.zeros((circ_gipsim_matrix.shape))
#     for i in range(circ_sim_matrix.shape[0]):
#         for j in range(circ_sim_matrix.shape[1]):
#             if circ_exp_sim[i,j] > 0:
#                 circ_sim_matrix[i,j] = circ_exp_sim[i,j]
#             else:
#                 circ_sim_matrix[i,j] = circ_gipsim_matrix[i,j]
#
#     return circ_sim_matrix

def compute_circ_sim(circ_gipsim_matrix):
    with h5py.File('../Data/circ_file/circRNA_expression_sim.h5', 'r') as hf:
        circ_exp_sim = hf['infor'][:]
    circ_sim_matrix = np.zeros((circ_gipsim_matrix.shape))
    for i in range(circ_sim_matrix.shape[0]):
        for j in range(circ_sim_matrix.shape[1]):
            if np.all(rel_matrix[i,:] == 0):
                # gip_mean = np.mean(circ_gipsim_matrix[:,j])
                # if gip_mean < circ_exp_sim[i,j]:
                circ_sim_matrix[i,:] = circ_exp_sim[i,:]
                # else:
                #     circ_sim_matrix[i,j] = gip_mean
            else:
                circ_sim_matrix[i,j] = circ_gipsim_matrix[i,j]

    return circ_sim_matrix, circ_exp_sim

# 读取h5文件
# 读取circRNA-disease之间的关系

with h5py.File('./disease-circRNA.h5', 'r') as f:
    circrna_disease_matrix = f['infor'][:]
    circrna_disease_matrix_val = circrna_disease_matrix.copy()

# 这里读取one_list
with h5py.File('./one_list_file/one_list.h5', 'r') as f:
    one_list = f['one_list'][:]
# 这里要把one_list还原为原来list的形式
temp_list = []
for temp in one_list:
    temp_list.append(tuple(temp))
one_list = temp_list
split = math.ceil(len(one_list) / 5)

# 创建五折交叉运算后要记录的数据结构
all_tpr = []
all_fpr = []
all_recall = []
all_precision = []
all_accuracy = []
all_F1 = []
fold=1

for i in range(0, len(one_list), split):
    # 根据每次fold的值不一样，读取特征的h5文件不一样
    with h5py.File('./circRNA_disease_gcn_embedding_32_feature_file/circRNA_disease_gcn_embedding_32_feature_fold%d.h5'%fold, 'r') as f:
        gcn_circRNA_feature = f['user_feature'][:]
        gcn_disease_feature = f['item_feature'][:]

    # 把一部分已知关系置零
    test_index = one_list[i:i+split]
    train_index = list(set(one_list)-set(test_index))

    new_circrna_disease_matrix = circrna_disease_matrix.copy()
    for index in test_index:
        new_circrna_disease_matrix[index[0], index[1]] = 0
    roc_circrna_disease_matrix = new_circrna_disease_matrix+circrna_disease_matrix
    rel_matrix = new_circrna_disease_matrix

    # 计算当前已知关系矩阵的高斯相似性
    makesimilaritymatrix = MakeSimilarityMatrix(rel_matrix)
    circ_gipsim_matrix, dis_gipsim_matrix = makesimilaritymatrix.circsimmatrix, makesimilaritymatrix.dissimmatrix

    fold += 1

    circ_sim_matrix, circ_exp_sim_matrix = compute_circ_sim(circ_gipsim_matrix)


    # 获取训练集和测试集
    input_fusion_feature_x = []
    input_fusion_x_label=[]
    for (u,i) in train_index:
        # 正样本
        gcn_circRNA_array = gcn_circRNA_feature[u,:]
        gcn_disease_array = gcn_disease_feature[i,:]
        fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
        input_fusion_feature_x.append(fusion_feature.tolist())
        input_fusion_x_label.append(1)
        # 计算当前circRNA与其相似的circRNA前十五个
        top_fif_circ_list = top_fif_sim_circ(circ_sim_matrix, u)
        associate_disease_set = set()
        associate_disease_set = find_associate_disease(top_fif_circ_list, rel_matrix, associate_disease_set)
        # 负样本
        for num in range(20):
            j = np.random.randint(89)
            while ((u,j) in train_index) or (j in associate_disease_set):
                j = np.random.randint(89)
            gcn_disease_array = gcn_disease_feature[j,:]
            fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
            input_fusion_feature_x.append(fusion_feature.tolist())
            input_fusion_x_label.append(0)

    input_fusion_feature_test_x=[]
    input_fusion_test_x_label=[]
    # 测试集构造，同样每个正样本选择一个负样本作为测试集
    for row in range(rel_matrix.shape[0]):
        for col in range(rel_matrix.shape[1]):
            gcn_circRNA_array = gcn_circRNA_feature[row, :]
            gcn_disease_array = gcn_disease_feature[col,:]
            fusion_feature = np.concatenate((gcn_circRNA_array, gcn_disease_array), axis=0)
            input_fusion_feature_test_x.append(fusion_feature.tolist())
            input_fusion_test_x_label.append(rel_matrix[row,col])

    # 构造神经网络
    model = Sequential()
    model.add(Dense(256, input_shape=(128,), W_regularizer=l2(0.0001), activation='relu', name='dense1'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.0001), name='dense2'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', W_regularizer=l2(0.0001), name='dense3'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', init='glorot_normal', name='prediction'))
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(np.array(input_fusion_feature_x), np.array(input_fusion_x_label), epochs=250, batch_size=100)
    predictions = model.predict(np.array(input_fusion_feature_test_x), batch_size=100)
    # 把这个预测prediction拟合为533*89的形式
    prediction_matrix = np.zeros((533, 89))
    predictions_index = 0
    for row in range(prediction_matrix.shape[0]):
        for col in range(prediction_matrix.shape[1]):
            prediction_matrix[row, col] = predictions[predictions_index]
            predictions_index += 1
    aa = prediction_matrix.shape
    bb = roc_circrna_disease_matrix.shape
    zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
    print(prediction_matrix.shape)
    print(roc_circrna_disease_matrix.shape)

    score_matrix_temp = prediction_matrix.copy()
    score_matrix = score_matrix_temp + zero_matrix
    minvalue = np.min(score_matrix)
    score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20
    sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix, roc_circrna_disease_matrix)

    tpr_list = []
    fpr_list = []
    recall_list = []
    precision_list = []
    accuracy_list = []
    F1_list = []
    for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
        P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
        N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
        TP = np.sum(P_matrix == 1)
        FP = np.sum(P_matrix == 0)
        TN = np.sum(N_matrix == 0)
        FN = np.sum(N_matrix == 1)
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        F1 = (2 * TP) / (2 * TP + FP + FN)
        F1_list.append(F1)

        accuracy_list.append(accuracy)
    all_tpr.append(tpr_list)
    all_fpr.append(fpr_list)
    all_recall.append(recall_list)
    all_precision.append(precision_list)
    all_accuracy.append(accuracy_list)
    all_F1.append(F1_list)

tpr_arr = np.array(all_tpr)
fpr_arr = np.array(all_fpr)
recall_arr = np.array(all_recall)
precision_arr = np.array(all_precision)
accuracy_arr = np.array(all_accuracy)
F1_arr = np.array(all_F1)

mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
mean_cross_fpr = np.mean(fpr_arr, axis=0)

mean_cross_recall = np.mean(recall_arr, axis=0)
mean_cross_precision = np.mean(precision_arr, axis=0)
mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
# 计算此次五折的平均评价指标数值
mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f"%(mean_accuracy, mean_recall, mean_precision, mean_F1))

roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
AUPR = np.trapz(mean_cross_precision, mean_cross_recall)
print(roc_auc)
print(AUPR)

with h5py.File('./IGNSCDA_final_32_k_20_AUC.h5') as hf:
    hf['fpr'] = mean_cross_fpr
    hf['tpr'] = mean_cross_tpr
# with h5py.File('./GCNMLPCDA_32d_AUPR.h5') as f:
#     f['recall'] = mean_cross_recall
#     f['precision'] = mean_cross_precision

plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0)
plt.savefig("roc-gcn-fold5-20.png")
print("runtime over, now is :")
# print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
plt.show()