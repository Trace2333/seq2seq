import logging
import os
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from evaluate_tools import rouge1, acc_metrics
from seq2seq_model import seq2seqBase, seqDataset
from data_precess import pickleRead, process, fileTolist, collate_fn, dictTondarray, evalcollate_fn


#process()    # 数据预处理

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = 'cuda:0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init()
wandb.login()

embwEN = dictTondarray(pickleRead(".\\embw\\ENembw.pkl"))
embwZH = dictTondarray(pickleRead(".\\embw\\ZHembw.pkl"))
trainEN = fileTolist(".\\cs_en\\train\\news-commentary-v13.zh-en.en")
trainZH = fileTolist(".\\cs_en\\train\\news-commentary-v13.zh-en.zh")
testEN = fileTolist(".\\cs_en\\test\\newstest2017.tc.en")
testZH = fileTolist(".\\cs_en\\test\\newstest2017.tc.zh")
embwEN = torch.tensor(embwEN, dtype=torch.float32).to(device)
embwZH = torch.tensor(embwZH, dtype=torch.float32).to(device)

epochs = 1
evalEpochs = 0
batchsize = 16
hiddensize = 300
inputsize = 300
lr = 2e-3

model = seq2seqBase(
    inputSize=inputsize,
    hiddenSize=hiddensize,
    batchSize=batchsize,
    numLayers=1,
    dictLen=len(embwZH),
    embwEN=embwEN,
    embwZH=embwZH
).to(device)

for i in model.modules():    # 参数初始化
    if isinstance(i, torch.nn.Linear):
        torch.nn.init.xavier_normal_(i.weight, gain=1)

dataset1 = seqDataset(    # 数据准备
    tokensListEN=trainEN,
    tokensListZH=trainZH
)
dataset2 = seqDataset(
    tokensListEN=testEN,
    tokensListZH=testZH
)
trainLoader = DataLoader(
    dataset=dataset1,
    shuffle=True,
    drop_last=True,
    batch_size=batchsize,
    num_workers=0,
    collate_fn=collate_fn
)
evalLoader = DataLoader(
    dataset=dataset2,
    shuffle=True,
    batch_size=batchsize,
    num_workers=0,
    collate_fn=evalcollate_fn
)

#model.load_state_dict(torch.load('seq2seq.pth'))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossfun = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    iteration = tqdm(trainLoader, desc=f"Training at epoch {epoch + 1}")
    model.train()
    for step, batch in enumerate(iteration):
        out = model(batch[0], batch[1])
        y = torch.tensor(batch[1], dtype=torch.long).to(device)
        acc = acc_metrics(out.argmax(2), y)
        out = out.permute(0, 2, 1)
        loss = lossfun(out, y) / batchsize
        rouge1 = 0

        wandb.log({"Acc:": acc})
        wandb.log({"Loss:": loss})
        for name, parms in model.named_parameters():
            wandb.log({f"{name} Weight:": torch.mean(parms.data)})
            if parms.grad is not None:    # 屏蔽掉embedding Parameter
                wandb.log({f"{name} Grad_Value:": torch.mean(parms.grad)})
        iteration.set_postfix(loss='{:.6f}'.format(loss.item()), rouge1='{:.3f}'.format(rouge1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), ".\\seq2seq.pth")

for epoch in range(evalEpochs):
    y = model.ZH[1]
    model.eval()
    iteration = tqdm(evalLoader, desc=f"Evaluation at epoch {epoch + 1}")
    for step, batch in enumerate(iteration):
        out, y = model(batch[0], y, ifEval=True)

