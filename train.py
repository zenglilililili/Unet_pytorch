import models.UnetModel as Models
from getData import get_train_data
from myloss import DiceLoss
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import torch
from myloss import softmax_mse_loss
from train_meanT import train_meanteacher

# 检查可用GPU
print("Let's use", torch.cuda.device_count(), "GPUs!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 设置参数
config = dict()
config["batchsize"] = 12
config["epochs"] = 1000
config["batch_epochs"] = 30
config["root"] = os.getcwd()  # 根目录
config["lr"] = 0.03  # 学习率
config["label_learning"] = True  # True表示监督学习
config["minloss"] = 20  # 记录最小验证集误差
config["val_split"] = 0.1

# 路径
x_data_path = "/data/CTV_base_CT.npy"
y_data_path = "/data/CTV_base_label.npy"
x_un_path = "/data/CTV_doc0_style_CT.npy"

# 加载数据
labeled_train_loader, labeled_val_loader, unlabeled_val_loader = get_train_data(x_data_path, y_data_path, x_un_path,
                                                                                config)

# Model
print("==> creating model")


def create_model(ema=False):
    models = Models.UNet(in_ch=1, out_ch=1)
    models = models.to(device)
    if ema:
        for param in models.parameters():
            param.detach_()  # 截断反向传播梯度流
    return models


model = create_model()  # 监督学习模型
ema_model = create_model(ema=True)  # 无监督学习模型

cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

criterion = DiceLoss()
consistency_criterion = softmax_mse_loss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# tensorboard 记录训练过程
writer = SummaryWriter(config["root"] + "/result/runs/1012")

start_epoch = 0

for epoch in range(start_epoch, config["epochs"]):
    minloss = train_meanteacher(labeled_train_loader, labeled_val_loader, unlabeled_val_loader, model,
                                ema_model, optimizer, criterion, consistency_criterion, writer, config, epoch, device)
    config["lr"] = config["lr"] - 0.00001
    writer.add_scalar('lr', config["lr"], epoch)
print('Finished Training')
writer.close()
torch.save(ema_model, config["root"] + '/endmodel.pkl')
