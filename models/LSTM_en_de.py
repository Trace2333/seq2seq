import torch
import random
import torch.nn as nn


class EncoderBase(nn.Module):
    """
    基本编码器
    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, device):
        super(EncoderBase, self).__init__()
        self.layers = numLayers
        self.batch = batchSize
        self.hiddenSize = hiddenSize
        self.recurent = nn.LSTM(inputSize, batch_first=True,
                                num_layers=numLayers, hidden_size=hiddenSize)  # Using a 2 direction RNN to read the input sentences
        self.device = device
        self.weight_init()

    def forward(self, x):
        """前向传播"""
        out, (hidden, cell) = self.recurent(x)
        return hidden, cell

    def initZeroState(self):
        """零初始化隐藏层"""
        return torch.zeros(self.layers, self.batch, self.hiddenSize).to(self.device)

    def initNormState(self):
        """正泰分布初始化隐藏层"""
        return torch.randn([self.layers, self.batch, self.hiddenSize]).to(self.device)

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)
            if isinstance(module, nn.LSTM):
                pass


class DecoderBase(nn.Module):
    """
    基本解码器

    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, dictLen, device):
        super(DecoderBase, self).__init__()
        """初始化模型"""
        self.layers = numLayers
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.batchSize = batchSize
        self.recurrent = nn.LSTM(inputSize, hiddenSize, bias=True)
        self.linear = nn.Linear(in_features=hiddenSize, out_features=dictLen)
        self.activation = nn.ReLU(inplace=False)
        self.device = device
        self.weight_init()


    def forward(self, x, in_hidden, in_cell):
        """前向计算"""
        out, (hidden, cell) = self.recurrent(x, (in_hidden, in_cell))
        out = out.permute(1, 0, 2)
        out = self.activation(out)
        y = self.linear(out)
        return y, hidden, cell

    def initZeroState(self):
        """隐藏层零初始化"""
        return torch.zeros(self.layers * 2, 2, self.hiddenSize).to(self.device)

    def initNormState(self):
        """隐藏层正态初始化"""
        return torch.randn([self.layers * 2, 2, self.hiddenSize]).to(self.device)

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1)
            if isinstance(module, nn.LSTM):
                pass


class Seq2seqBase(nn.Module):
    """
    基本seq2seq模型
    利用EncoderBase和DecoderBase搭建

    """
    def __init__(self, inputSize, hiddenSize, batchSize, numLayers, dictLen, embwEN, embwZH, device, start_TF_rate):
        """层初始化"""
        super(Seq2seqBase, self).__init__()
        self.encoder = EncoderBase(inputSize, hiddenSize, batchSize, numLayers, device)
        self.decoder = DecoderBase(hiddenSize, hiddenSize, batchSize, numLayers, dictLen, device)  # 取最后一个输入的隐层状态作为语义向量
        self.EN = nn.Parameter(embwEN)
        self.ZH = nn.Parameter(embwZH)
        self.batchsize = batchSize
        self.BN = nn.BatchNorm1d(300)
        self.start_TF_rate = start_TF_rate
        self.device = device

    def forward(self, x, y):
        """前向计算"""
        x = nn.functional.embedding(torch.tensor(x).long().to(self.device), self.EN)
        y = nn.functional.embedding(torch.tensor(y).long().to(self.device), self.ZH)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        y = y.permute(1, 0, 2).split(1, dim=0)
        hidden, cell = self.encoder(x)
        out, hidden, cell = self.decoder(y[0], hidden, cell)
        for i in y[1:]:
            if self.start_TF_rate > random.uniform(0, 1):    # All teacher forcing
                p, hidden, cell = self.decoder(i, hidden, cell)
            else:
                decode_in = nn.functional.embedding(out.split(1, dim=1)[-1].argmax(2), self.ZH).permute(1, 0, 2)
                p, hidden, cell = self.decoder(decode_in, hidden, cell)
            out = torch.cat((out, p), dim=1)
        return out

    def evaluation(self, x):
        x = nn.functional.embedding(torch.tensor(x).long().to(self.device), self.EN)
        x = x.to(torch.float32)
        out = torch.full([self.batchsize, 1], 1).to(self.device)
        eos = torch.tensor(len(self.ZH) - 1).to(self.device)
        hidden, cell = self.encoder(x)  # 初始语义向量
        pred_input = torch.full([self.batchsize, 1], 1).to(self.device)
        while torch.equal(out.split(1, dim=1)[-1][0], eos) is False and out.size()[1] != x.size()[1]:
            pred_input = nn.functional.embedding(pred_input, self.ZH)
            pred_input, hidden, cell = self.decoder(pred_input.permute(1, 0, 2), hidden, cell)
            pred_input = pred_input.argmax(2)
            out = torch.cat((out, pred_input), dim=1)
        return out


