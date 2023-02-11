import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import geniter, geniter1
import HCTnet
import Utils


def loadData():
    # 读入数据
    data_HSI = sio.loadmat('data/HSI_Trento.mat')['hsi_trento']
    data_lidar = sio.loadmat('data/Lidar1_Trento.mat')['lidar1_trento']
    labels = sio.loadmat('data/GT_Trento.mat')['gt_trento']

    return data_HSI, data_lidar, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def sampling1(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m+1):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m+1):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def select_traintest(groundTruth):  #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = int(max(groundTruth))
    amount = [
            129, 125, 105, 154, 184, 122,
        ]
    # amount = [
    #     50, 50, 50, 50, 50, 50,
    # ]
    for i in range(m):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1
        ]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

BATCH_SIZE = 64

def create_data_loader():
    # 地物类别
    # class_num = 6
    # 读入数据
    X1, X2, y = loadData()
    # 每个像素周围提取 patch 的尺寸
    patch_size = 11
    PATCH_LENGTH = int((patch_size - 1) / 2)
    TOTAL_SIZE = 30214
    ALL_SIZE = 99600
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    print('Hyperspectral data shape: ', X1.shape)
    print('Lidar data shape: ', X2.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X1 = applyPCA(X1, numComponents=pca_components)
    print('Data shape after PCA: ', X1.shape)

    # 将数据变换维度： [m,n,k]->[m*n,k]
    X1_all_data = X1.reshape(np.prod(X1.shape[:2]), np.prod(X1.shape[2:]))
    X2_all_data = X2.reshape(np.prod(X2.shape[:2]),)
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(np.int)
    CLASSES_NUM = max(gt)
    print(CLASSES_NUM)

    # 数据标准化
    X1_all_data = preprocessing.scale(X1_all_data)
    data_X1 = X1_all_data.reshape(X1.shape[0], X1.shape[1], X1.shape[2])
    whole_data_X1 = data_X1
    padded_data_X1 = np.lib.pad(whole_data_X1, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)), 'constant', constant_values=0)

    X2_all_data = preprocessing.scale(X2_all_data)
    data_X2 = X2_all_data.reshape(X2.shape[0], X2.shape[1])
    whole_data_X2 = data_X2
    padded_data_X2 = np.lib.pad(whole_data_X2, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH)),
                             'constant', constant_values=0)
    print('\n... ... create train & test data ... ...')
    train_indices, test_indices = select_traintest(gt)
    # train_indices, test_indices = sampling(0.99, gt)
    _, all_indices = sampling1(1, gt)
    _, total_indices = sampling(1, gt)
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)

    print('\n-----Selecting Small Cube from the Original Cube Data-----')
    train_iter, test_iter, total_iter, all_iter = geniter.generate_iter(
        TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
        ALL_SIZE, all_indices, whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
        pca_components, BATCH_SIZE, gt)
    # train_iter, test_iter = geniter1.generate_iter(
    #     TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
    #     whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
    #     pca_components, BATCH_SIZE, gt)

    return train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices
    # return train_iter, test_iter,


def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = HCTnet.HCTnet().to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data1, data2, target) in enumerate(train_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data1, data2)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for (data1, data2, labels) in test_loader:
        data1, data2 = data1.to(device), data2.to(device)
        outputs = net(data1, data2)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Apple Tree', 'Building', 'Ground', 'Wood', 'Vineyard', 'Roads']

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':
    for i in range(1):
        train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices = create_data_loader()
        # train_iter, test_iter = create_data_loader()
        tic1 = time.perf_counter()
        net, device = train(train_iter, epochs=100)
        # 只保存模型参数
        torch.save(net.state_dict(), 'cls_params/Trento_params01.pth')
        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_iter)
        toc2 = time.perf_counter()
        # 评价指标b
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
        classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2
        file_name = "cls_result/HCT_100epochs_819_tr_classification_report{}.txt".format(i)
        with open(file_name, 'w') as x_file:
            x_file.write('{} Training_Time (s)'.format(Training_Time))
            x_file.write('\n')
            x_file.write('{} Test_time (s)'.format(Test_time))
            x_file.write('\n')
            x_file.write('{} Kappa accuracy (%)'.format(kappa))
            x_file.write('\n')
            x_file.write('{} Overall accuracy (%)'.format(oa))
            x_file.write('\n')
            x_file.write('{} Average accuracy (%)'.format(aa))
            x_file.write('\n')
            x_file.write('{} Each accuracy (%)'.format(each_acc))
            x_file.write('\n')
            x_file.write('{}'.format(classification))
            x_file.write('\n')
            x_file.write('{}'.format(confusion))
        print('------Get classification results successful-------')

        # Utils.generate_png(
        #     total_iter, net, y, device, total_indices, 'cls_map/' + 'Trento_{}'.format(i))
        # Utils.generate_all_png(
        #     all_iter, net, y, device, all_indices, 'cls_map/' + 'Trento_all_{}'.format(i))






