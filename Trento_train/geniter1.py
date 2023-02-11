import torch
import numpy as np
import torch.utils.data as Data

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic_1(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def select_small_cubic_2(data_size, data_indices, whole_data, patch_length, padded_data):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
                  whole_data1, whole_data2, PATCH_LENGTH, padded_data1, padded_data2, INPUT_DIMENSION, batch_size, gt):
    # gt_all = gt[all_indices] - 1
    # gt_total = gt[total_indices]-1
    y_train = gt[train_indices]-1
    y_test = gt[test_indices]-1

    # X1_total_data = select_small_cubic_1(TOTAL_SIZE, total_indices, whole_data1, PATCH_LENGTH, padded_data1, INPUT_DIMENSION)
    # print(X1_total_data.shape)
    # X2_total_data = select_small_cubic_2(TOTAL_SIZE, total_indices, whole_data2, PATCH_LENGTH, padded_data2)
    # print(X2_total_data.shape)

    # X1_all_data = select_small_cubic_1(ALL_SIZE, all_indices, whole_data1, PATCH_LENGTH, padded_data1, INPUT_DIMENSION)
    # print(X1_all_data.shape)
    # X2_all_data = select_small_cubic_2(ALL_SIZE, all_indices, whole_data2, PATCH_LENGTH, padded_data2)
    # print(X2_all_data.shape)

    X1_train_data = select_small_cubic_1(TRAIN_SIZE, train_indices, whole_data1, PATCH_LENGTH, padded_data1, INPUT_DIMENSION)
    print(X1_train_data.shape)
    X2_train_data = select_small_cubic_2(TRAIN_SIZE, train_indices, whole_data2, PATCH_LENGTH, padded_data2)
    print(X2_train_data.shape)

    X1_test_data = select_small_cubic_1(TEST_SIZE, test_indices, whole_data1, PATCH_LENGTH, padded_data1, INPUT_DIMENSION)
    print(X1_test_data.shape)
    X2_test_data = select_small_cubic_2(TEST_SIZE, test_indices, whole_data2, PATCH_LENGTH, padded_data2)
    print(X2_test_data.shape)

    X1_train = X1_train_data.transpose(0, 3, 1, 2)
    X1_test = X1_test_data.transpose(0, 3, 1, 2)
    print('after transpose: Xtrain shape: ', X1_train.shape)
    print('after transpose: Xtest  shape: ', X1_test.shape)

    x1_tensor_train = torch.from_numpy(X1_train).type(torch.FloatTensor).unsqueeze(1)
    x2_tensor_train = torch.from_numpy(X2_train_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, x2_tensor_train, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(X1_test).type(torch.FloatTensor).unsqueeze(1)
    x2_tensor_test = torch.from_numpy(X2_test_data).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.LongTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, x2_tensor_test, y1_tensor_test)

    # X1_total = X1_total_data.transpose(0, 3, 1, 2)
    # X1_all = X1_all_data.transpose(0, 3, 1, 2)

    # X1_total_tensor_data = torch.from_numpy(X1_total).type(torch.FloatTensor).unsqueeze(1)
    # X2_total_tensor_data = torch.from_numpy(X2_total_data).type(torch.FloatTensor).unsqueeze(1)
    # total_tensor_data_label = torch.from_numpy(gt_total).type(torch.LongTensor)
    # torch_dataset_total = Data.TensorDataset(X1_total_tensor_data, X2_total_tensor_data, total_tensor_data_label)

    # X1_all_tensor_data = torch.from_numpy(X1_all).type(torch.FloatTensor).unsqueeze(1)
    # X2_all_tensor_data = torch.from_numpy(X2_all_data).type(torch.FloatTensor).unsqueeze(1)
    # all_tensor_data_label = torch.from_numpy(gt_all).type(torch.LongTensor)
    # torch_dataset_all = Data.TensorDataset(X1_all_tensor_data, X2_all_tensor_data)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    # total_iter = Data.DataLoader(
    #     dataset=torch_dataset_total,  # torch TensorDataset format
    #     batch_size=batch_size,  # mini batch size
    #     shuffle=False,
    #     num_workers=0,
    # )
    # all_iter = Data.DataLoader(
    #     dataset=torch_dataset_all,  # torch TensorDataset format
    #     batch_size=batch_size,  # mini batch size
    #     shuffle=False,
    #     num_workers=0,
    # )
    return train_iter, test_iter,  #, y_test
