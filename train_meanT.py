import torch
import time


def train_meanteacher(labeled_train_trainloader, labeled_val_trainloader, unlabeled_val_trainloader, model,
                      ema_model, optimizer, criterion, consistency_criterion, writer, config, epoch, device):
    model.train()
    model.to(device)

    start = time.time()

    loss = 0

    labeled_train_iter = iter(labeled_train_trainloader)

    for batch_idx in range(config["batch_epochs"]):
        try:
            inputs_x, target_x = next(labeled_train_iter)
        except StopIteration:
            continue
        inputs_x, target_x = inputs_x.to(device), target_x.to(device)
        logits_x = model(inputs_x)
        # tensorboard  记录网络结构
        if epoch == 0:
            writer.add_graph(model, (logits_x,))

        logits_x = logits_x.flatten()
        targets_x = target_x.flatten()
        lx = criterion(logits_x, targets_x)  # 监督学习loss

        loss = lx

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证集
    loss_val = 0
    intput_val_x_iter = iter(labeled_val_trainloader)
    for val_i in range(len(labeled_val_trainloader)):
        with torch.no_grad():
            try:
                intput_val_x, target_val_x = next(intput_val_x_iter)
            except StopIteration:
                continue

            intput_val_x, target_val_x = intput_val_x.to(device), target_val_x.to(device)
            val_x = model(intput_val_x)
            val_x = val_x.flatten()
            target_val_x = target_val_x.flatten()

        loss_val = loss_val + criterion(val_x, target_val_x).item()

    end = time.time()
    loss_val = loss_val / len(labeled_val_trainloader)
    print('epoc {:}  time {:}s  loss: {:.6f}  val_loss: {:.6f}'.format(epoch, int(end - start), loss, loss_val))
    if loss_val < config["minloss"]:
        config["minloss"] = loss_val
        torch.save(model,
                   config["root"] + '/result/bestModel/epoch' + str(epoch) + "_" + str(loss_val) + '_min_model.pkl')
        print(
            '\n save model to ' + config["root"] + '/result/bestModel/epoch' + str(epoch) + "_" + str(
                loss_val) + '_min_model.pkl')

    writer.add_scalar('losses/train_loss', loss, epoch)
    writer.add_scalar('losses/val_loss', loss_val, epoch)
