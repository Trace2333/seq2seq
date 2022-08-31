import torch
import numpy as np
from data_precess import pickleRead, dictCreate, fileTolist

"""from transformers import BertModel
T1 = torch.randn([4, 10, 30])
#For the dimension -2
T2 = T1[-2, :, :]
T3 = T1[-1, : ,:]
print("第一个维度按列表方式取-2 :", T2.size())
#结果表示按照标定取法得到的是最后一个隐藏层的输出，而整体Tensor是所有隐藏层的输出连接，即得到正向RNN上的最后一个词输入得到的隐藏层状态
print("原Tensor：", T1.size())
print("第一个维度按列表方式取-1 ：", T3.size())
#得到的是反向RNN中的第一个词输入的时候的隐藏层状态
"""

"""inputs = torch.randn([16, 20, 300])
rnn = torch.nn.RNN(300, 200, bidirectional=True, batch_first=True)
out = rnn(inputs)
#without the hiddenstate, default is ZERO
print("out:", out)
#print("out1_SIZE:", out1.size())
#Print wrong! output is a tuple
print("out1:", out[0])
print("out2:", out[1])
print("output_SIZE:", out[0].size())
print("hiddensize_SIZE:", out[1].size())
"""
"""embwZH = pickleRead(".\\embw\\ZHembw.pkl")
no = np.empty(shape=(0, 300))
for i in embwZH:
    no.append(i)
print(no)"""
"""T = torch.randn([1, 1, 10])
F = torch.randn([1, 1, 10])
T = T.argmax(2).squeeze(0)
F = F.argmax(2).squeeze(0)

if T == 0:
    print("True")
print(T)
print(F)
cated = torch.cat((T, F), dim=0).unsqueeze(0)
print(cated)
emp = torch.empty([1, 1]).squeeze(0)
print(emp)
print(T)"""

"""dataset = pickleRead(".\\datasets\\trainZH.pkl")
for i in dataset:
    for j, s in zip(i, range(10)):
        print(j)"""
"""
filenames = [
        ".\\cs_en\\train\\news-commentary-v13.zh-en.en",
        ".\\cs_en\\train\\news-commentary-v13.zh-en.zh",
        ".\\cs_en\\test\\newstest2017.tc.en",
        ".\\cs_en\\test\\newstest2017.tc.zh",
        ".\\cs_en\\dev\\newsdev2017.tc.zh",
        ".\\cs_en\\dev\\newsdev2017.tc.en"
    ]

#dictCreate(fileTolist(filenames[0]), "Endataset.pkl", fileTolist(filenames[2]), fileTolist(filenames[5]))
dataset = pickleRead(".\\datasets\\ENdataset.pkl")
print(dataset)"""
"""
A = torch.randn([32,30,300])
B = A.permute(1, 0, 2)
print(B.size())
C = torch.split(A, 8, dim=1)
print("A在维度1上分离一个元素", C[0].size())
#print("A剩余:", C[1].size())
for i in C[:]:
        print(i.size())"""


"""A = torch.randn([32, 30, 300])
print(A[-1].size())
print((A.split(1, dim=1)[-1]).argmax(2))"""


"""class argparse():
    pass
argparse.seed = 42
print(argparse)
print(argparse.seed)"""

import torch.nn as nn
"""rnncell = nn.RNNCell(input_size=300, hidden_size=300, nonlinearity='tanh', bias=True)
# 所有的四个参数,没有batch_first
x = torch.randn([16, 30, 300])
x = x.permute(1, 0, 2)
# x-->【length， batch， vecsize】， [30, 16, 300]
hidden = torch.randn([16, 300])
out = rnncell(x[0], hidden)
print(out.size())
print(x[0].size())"""

"""
inputs = torch.randn([16, 300])
norm_in = torch.randn([16, 1, 300])
norm_in = norm_in.permute(0, 2, 1)
# 输入的数据模拟
lstmcell = nn.LSTMCell(input_size=300, hidden_size=300)
norm = nn.BatchNorm1d(300)
norm_out = norm(norm_in)
print("NormOu size::", norm_out.size())
out, hidden = lstmcell(inputs)
print("OUT size:", out.size())
print("HIden size", hidden.size())
lstm = nn.LSTM(input_size=300, hidden_size=300, bidirectional=True, batch_first=True)
inputs2 = torch.randn([16, 30, 300])
out, hidden = lstm(inputs2)
print("LSTM out :", out.size())
print("LSTM Hidden size:", hidden[1].size())"""

"""inputs = torch.randn([16, 30, 300])
out1 = inputs.split(1, dim=1)
s = 0
for i in out1[1:]:
    s += 1
print(s)
out2 = inputs.split(inputs.size()[1] - 1, dim=1)
for i in out2:
    print(i.size())"""
"""
norm_in = torch.randn([16, 1, 300])
norm_in = norm_in.permute(0, 2, 1)
norm = nn.BatchNorm1d(300)
norm_out = norm(norm_in)
print("NormOu size::", norm_out.size())
"""
"""
import random

generator = random.random()
a = generator    # 作为集束搜索的概率生成器


"""


"""filenames = [
        ".\\cs_en\\train\\news-commentary-v13.zh-en.en",
        ".\\cs_en\\train\\news-commentary-v13.zh-en.zh",
        ".\\cs_en\\test\\newstest2017.tc.en",
        ".\\cs_en\\test\\newstest2017.tc.zh",
        ".\\cs_en\\dev\\newsdev2017.tc.zh",
        ".\\cs_en\\dev\\newsdev2017.tc.en"
    ]

#dictCreate(fileTolist(filenames[0]), "Endataset.pkl", fileTolist(filenames[2]), fileTolist(filenames[5]))
dataset1 = pickleRead(".\\datasets\\ENdataset.pkl")
dataset2 = pickleRead(".\\datasets\\ZHdataset.pkl")
print(dataset1)"""

import heapq

a = [[1, 2, 3], [2, 3, 4]]
b = [[1, 2, 3], [2, 3, 4]]
c = a + b
d = []
for i, j in zip(a, b):
    s = i + j
    d.append(s)
print(d)