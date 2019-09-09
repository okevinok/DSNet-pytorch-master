import torch
import os
from torch.autograd import Variable

"""
这里展现了训练的具体信息
"""

def train_network(model, cfg, train_loader, optimizer, epoch): # TODO 实际使用的训练函数
    print("调用的时候执行train")
    # model.train()  这个应该是不需要的
    total_loss = 0

    print (len(optimizer.param_groups))
    for batch_idx, sampled_batch in enumerate(train_loader):
        optimizer.zero_grad()  # 将所有梯度设为0
        sample_data = sampled_batch[0].reshape(sampled_batch[0].shape[0],sampled_batch[0].shape[3], sampled_batch[0].shape[1],sampled_batch[0].shape[2]).to(cfg.device)
        sample_data = Variable(sample_data.type(torch.cuda.FloatTensor))
        Y_true = Variable(sampled_batch[1].reshape(sampled_batch[1].shape[0],sampled_batch[1].shape[3], sampled_batch[1].shape[1],sampled_batch[1].shape[2]).to(cfg.device).type(torch.cuda.FloatTensor))

        nn_outputs = model(sample_data)

        # plt.imshow(sample_data.reshape(sample_data.shape[1], sample_data.shape[2], sample_data.shape[3]))
        # plt.show()
        #
        # plt.imshow(nn_outputs.reshape(nn_outputs.shape[1], nn_outputs.shape[2], nn_outputs.shape[3]))
        # plt.show()
        #
        # plt.imshow(nn_outputs.reshape(Y_true.shape[1], Y_true.shape[2], Y_true.shape[3]))
        # plt.show()

        loss = model.compute_loss(nn_outputs,Y_true , cfg)

        loss.backward()
        optimizer.step()

        total_loss += loss
        if batch_idx % cfg.log_interval == 0:  # TODO 训练过程中每100个batch进行一次输出
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * cfg.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_path,"ChauffeurNet_{}_{}.pt".format(epoch,batch_idx))) # TODO 存储模型

    total_loss /= len(train_loader)  # 求平均误差
    cfg.scheduler.step(total_loss) # 调整学习率 当loss不在下降的时候
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    del total_loss
