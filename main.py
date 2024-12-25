import random
import sys

sys.path.append('./function/')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from function.LKNetwheat import LKNet
from function.DeepSEA_model import DeepSEA
from function.data import train_loader, val_loader, n_train, n_val
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import datetime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False

setup_seed(168)

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#以上代码允许多进程进行

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# net = Net().to(device)
# net = Logistic().to(device)
# net = NetCnn(32336).to(device)
net=LKNet(32336,1).to(device)
# net=DeepSEA(32336,1).to(device)

Epoch=60
# net.load_state_dict(torch.load('LKNET.pkl'))
curr_time = datetime.datetime.now()
time = datetime.datetime.strftime(curr_time, '%Y-%m-%d_%H-%M-%S')

optimizer=optim.Adam(net.parameters(),lr=0.0001)

loss_func = nn.MSELoss()
# val_acc_all = []
# train_acc_all = []
train_loss_all = []
val_loss_all = []

train_pc_all=[]
test_pc_all=[]
train_pc_v_all=[]
test_pc_v_all=[]

fin_pc=[]
best_per=0.0
for epoch in range(Epoch):
    net.train()
    train_loss = 0.0

    train_pred_epoch=[]
    train_exp_epoch=[]
    for i, (exp, seq) in enumerate(train_loader, 0): #遍历可迭代对象，并返回索引与元素。
            # print(exp, seq, gene_name, gene_seq, flush=True)
            # print(seq.shape, flush=True)
        seq = seq.to(device)
        exp = exp.to(device)
        predict = net(seq)
        predict = predict.squeeze(-1)
        # print(predict, flush=True)
        # print(exp,flush=True)
            # pred_lab = torch.argmax(predict, 1)
        loss = loss_func(predict, exp.float())

        train_pred_epoch.extend(list(predict.cpu().detach().numpy()))
        train_exp_epoch.extend(list(exp.cpu().detach().numpy()))

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pc = pearsonr(train_pred_epoch,train_exp_epoch)
    train_pc_all.append(pc[0])
    train_pc_v_all.append(pc[1])   
    train_loss_all.append(train_loss /len(train_loader))

    print('epoch train loss  :', epoch, train_loss/len(train_loader), flush=True)  #实时打印
    print(f"--> Pearson: {pc[0]:.4f}, P-value: {pc[1]:.4f}", flush=True)
    print('===========================================================')

    net.eval()
    val_loss = 0.0

     # test
    test_pred_epoch=[]
    test_exp_epoch=[]
    with torch.no_grad():   
      for i, (exp, seq) in enumerate(val_loader, 0):            
            seq = seq.to(device)
            exp = exp.to(device)
            predict = net(seq)
            predict = predict.squeeze(-1)
            loss = loss_func(predict,exp.float())
            val_loss += loss.item()

            test_pred_epoch.extend(list(predict.cpu().detach().numpy()))
            test_exp_epoch.extend(list(exp.cpu().detach().numpy()))
 
    pc = pearsonr(test_pred_epoch,test_exp_epoch)
        # 取出最后一次的test的皮尔森系数保存下来
    if(epoch==Epoch-1):
        fin_pc=pc

    if pc[0]>best_per:
        best_per=pc[0]

    test_pc_all.append(pc[0])
    test_pc_v_all.append(pc[1])
    val_loss_all.append(val_loss/len(val_loader))
    print('epoch test loss  :',epoch, val_loss/len(val_loader), flush=True)
    print(f"--> Pearson: {pc[0]:.4f}, P-value: {pc[1]:.4f}", flush=True)
    print('===========================================================')

# plot
# torch.save(net.state_dict(),'LKNET.pkl')
# torch.save(net.state_dict(),'DEEPSEA.pkl')
print('end:',fin_pc,'best:',best_per,flush=True)
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(train_loss_all, 'ro-', label="Train Loss")
plt.plot(val_loss_all, 'bs-', label="Val Loss")
# 设置纵轴刻度的大小
plt.tick_params(axis='y', labelsize=30)
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_pc_all, 'ro-', label="Train PC")
plt.plot(test_pc_all, 'bs-', label="Val PC")
plt.legend()
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("pearsonr")


# plt.subplot(2, 2, 4)
# plt.plot(train_pc_v_all, 'ro-', label="Train PC")
# plt.plot(test_pc_v_all, 'bs-', label="Val PC")
# plt.legend()
# plt.grid()
# plt.xlabel("Epoch")
# plt.ylabel("pearsonr_value")
plt.savefig("./results/"+time+"-pearsonr.png")
plt.show()
    
#     plt.close()