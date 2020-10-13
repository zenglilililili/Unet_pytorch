import numpy as np
import torch
import torch.utils.data as data
import random
from sklearn.model_selection import train_test_split
from torch.autograd import Variable


def get_train_data(x_data_path, y_data_path, x_un_path, config):
    # 加载数据
    x_base_train = np.load(config["root"] + x_data_path, allow_pickle=True)
    y_base_train = np.load(config["root"] + y_data_path, allow_pickle=True)
    x_un_train = np.load(config["root"] + x_un_path, allow_pickle=True)

    shape = x_base_train.shape[0]
    un_shape = x_un_train.shape[0]

    x_train = x_base_train.reshape([shape, 1, 256, 256])
    y_train = y_base_train.reshape([shape, 1, 256, 256])
    x_un_train = x_un_train.reshape([un_shape, 1, 256, 256])

    index = [i for i in range(shape)]
    index_un = [i for i in range(un_shape)]

    random.seed(20)
    random.shuffle(index)
    random.shuffle(index_un)
    x_train = x_train[index]
    y_train = y_train[index]
    x_un_train = x_un_train[index_un]

    # split输入的数据集必须转换成numpy类型(只能处理numpy类型的数据)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config["val_split"])

    x_train = torch.from_numpy(x_train)  # numpy 转成 torch 类型
    x_train = x_train.type(torch.FloatTensor)

    x_val = torch.from_numpy(x_val)
    x_val = x_val.type(torch.FloatTensor)  # 转Float

    y_train = torch.from_numpy(y_train)
    y_train = y_train.type(torch.FloatTensor)  # 转Float

    y_val = torch.from_numpy(y_val)
    y_val = y_val.type(torch.FloatTensor)  # 转Float

    x_un_train = torch.from_numpy(x_un_train)
    x_un_train = x_un_train.type(torch.FloatTensor)  # 转Float

    x_train = Variable(x_train, requires_grad=True)
    y_train = Variable(y_train, requires_grad=True)
    x_val = Variable(x_val, requires_grad=True)
    y_val = Variable(y_val, requires_grad=True)
    x_un_train = Variable(x_un_train, requires_grad=True)

    # print(config["batchsize"])

    # dataloader
    # 先使用Data.TensorDataset(X_train, y_train) 封装起来，再使用Data.DataLoader()调用
    # 有标签的训练数据集 train
    labeled_train_dataset = data.TensorDataset(x_train, y_train)
    labeled_train_loader = data.DataLoader(dataset=labeled_train_dataset, batch_size=config["batchsize"], shuffle=False,
                                           drop_last=True)

    # 有标签的验证数据集 val
    labeled_val_dataset = data.TensorDataset(x_val, y_val)
    labeled_val_loader = data.DataLoader(dataset=labeled_val_dataset, batch_size=config["batchsize"], shuffle=False,
                                         drop_last=True)

    # 无标签的数据集
    unlabeled_dataset = data.TensorDataset(x_un_train, x_un_train)
    unlabeled_loader = data.DataLoader(dataset=unlabeled_dataset, batch_size=config["batchsize"], shuffle=False,
                                       drop_last=True)

    return labeled_train_loader, labeled_val_loader, unlabeled_loader