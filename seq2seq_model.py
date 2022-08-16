import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class encoderBase(nn.Module):
    """
    基本编码器
    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers):
        super(encoderBase, self).__init__()
        self.layers = numLayers
        self.batch = batchSize
        self.hiddenSize = hiddenSize
        self.recurent = nn.RNN(inputSize, batch_first=True, bidirectional=False,
                               num_layers=numLayers, hidden_size=hiddenSize)  # Using a 2 direction RNN to read the input sentences

    def forward(self, x):
        """前向传播"""
        state = self.initNormState()
        out = self.recurent(x, state)
        return out

    def initZeroState(self):
        """零初始化隐藏层"""
        return torch.zeros(self.layers, self.batch, self.hiddenSize).to(device)

    def initNormState(self):
        """正泰分布初始化隐藏层"""
        return torch.randn([self.layers, self.batch, self.hiddenSize]).to(device)


class decoderBase(nn.Module):
    """
    基本解码器

    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, dictLen):
        super(decoderBase, self).__init__()
        """初始化模型"""
        self.layers = numLayers
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.batchSize = batchSize
        self.recurrent = nn.RNNCell(inputSize, hiddenSize, nonlinearity='tanh', bias=True)
        self.linear = nn.Linear(in_features=hiddenSize, out_features=dictLen)

    def forward(self, x, state):
        """前向计算"""
        out = self.recurrent(x, state)
        state = out
        y = self.linear(out.unsqueeze(0).permute(1, 0, 2))
        return y, state

    def initZeroState(self):
        """隐藏层零初始化"""
        return torch.zeros(self.layers * 2, 2, self.hiddenSize).to(device)

    def initNormState(self):
        """隐藏层正态初始化"""
        return torch.randn([self.layers * 2, 2, self.hiddenSize]).to(device)


class seq2seqBase(nn.Module):
    """
    基本seq2seq模型
    利用EncoderBase和DecoderBase搭建

    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, dictLen, embwEN, embwZH):
        """层初始化"""
        super(seq2seqBase, self).__init__()
        self.encoder = encoderBase(inputSize, hiddenSize, batchSize, numLayers)
        self.decoder = decoderBase(hiddenSize, hiddenSize, batchSize, numLayers, dictLen)  # 取最后一个输入的隐层状态作为语义向量
        self.EN = nn.Parameter(embwEN)
        self.ZH = nn.Parameter(embwZH)
        self.batchsize = batchSize

    def forward(self, x, y, ifEval=False, start_TF_rate=0.5):
        """前向计算"""
        if ifEval is not True:
            x = nn.functional.embedding(torch.tensor(x).long().to(device), self.EN)
            y = nn.functional.embedding(torch.tensor(y).long().to(device), self.ZH)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            y_in = y.permute(1, 0, 2)
            output1, c = self.encoder(x)
            out, hidden = self.decoder(y_in[0], c)
            for i in y_in[1:]:
                if start_TF_rate > random.uniform(0, 1):    # All teacher forcing
                    p, hidden = self.decoder(i, c)
                else:
                    decode_in = nn.functional.embedding(out.split(1, dim=1)[-1].argmax(2), self.ZH).squeeze(1)
                    p, hidden = self.decoder(decode_in, c)
                out = torch.cat((out, p), dim=1)
            return out
        if ifEval:
            x = nn.functional.embedding(torch.tensor(x).long().to(device), self.EN)
            x = x.to(torch.float32)
            y = y.to(torch.float32).unsqueeze(0)
            out = torch.full([self.batchsize, 1], 1).to(device)
            EOS = torch.tensor(len(self.ZH) - 1).to(device)
            for i in range(31):
                y = torch.cat((y, self.EN[1].unsqueeze(0)), dim=0)
            y = y.reshape([self.batchsize, 1, 300])
            while not torch.equal(y[0].argmax(1), EOS):
                if out.size(1) == 1:
                    output1, c = self.encoder(x)
                y, c = self.decoder(y, c)
                out = torch.cat((out, y.argmax(2)), dim=1)
                y = nn.functional.embedding(y.argmax(2), self.ZH)
            return out


class seqDataset(Dataset):
    def __init__(self, tokensListEN, tokensListZH):
        """初始化，为其添加各种必要属性

            Args:
                tokensListEN:经过fileTolist打开后的返回列表,语言为英文
                tokensListZH:经过fileTolist打开后返回的列表，语言为中文
        """
        super(seqDataset, self).__init__()
        self.EN = tokensListEN
        self.ZH = tokensListZH

    def __getitem__(self, item):
        """
        处理句子，返回一个输出和它的index

        item:
        Return:
            一个列表和一个y（列表）,两项数据都是字符串类型的
        """
        return self.EN[item], self.ZH[item]    # 准备的是英文->中文的训练数据

    def __len__(self):
        """
        返回数据集长度

        Return:
            数据集长度
        """
        return len(self.EN)