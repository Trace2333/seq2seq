import sys
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append("..")
from utils.args import get_parameter
from utils.config_fun import load_config
from models.LSTM_en_de import Seq2seqBase
from utils.evaluate_tools import acc_metrics
from dataset.dataset_base import DatasetBase, collate_fn, evalcollate_fn
from utils.data_process import dictTondarray, pickleRead, fileTolist
#sys.path.clear()


def trainer_base(args=None):
    if args is None:
        args = get_parameter()   # 接收命令行参数
    wandb_config = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        evaluation_epoch=args.evaluation_epochs,
        optimizer=args.optimizer,
        lossfun=args.lossfun,
        model_name=args.model_name
    )
    wandb.login(
        host="http://47.108.152.202:8080",
        key="local-86eb7fd9098b0b6aa0e6ddd886a989e62b6075f0"
    )
    wandb.init(
        config=wandb_config,
        project=args.project,
        notes=args.notes
    )
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("Cuda disable! Using CPU for training.")
            device = torch.device('cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        print("Device unassigned...Using Cpu for training.")
        device = torch.device('cpu')

    embwEN = dictTondarray(pickleRead("./hot_data/embw/ENembw.pkl"))
    embwZH = dictTondarray(pickleRead("./hot_data/embw/ZHembw.pkl"))
    trainEN = fileTolist("./origin_data/cs_en/train/news-commentary-v13.zh-en.en")
    trainZH = fileTolist("./origin_data/cs_en/train/news-commentary-v13.zh-en.zh")
    testEN = fileTolist("./origin_data/cs_en/test/newstest2017.tc.en")
    testZH = fileTolist("./origin_data/cs_en/test/newstest2017.tc.zh")
    embwEN = torch.tensor(embwEN, dtype=torch.float32).to(device)
    embwZH = torch.tensor(embwZH, dtype=torch.float32).to(device)

    model = Seq2seqBase(
        inputSize=args.input_size,
        hiddenSize=args.hidden_size,
        batchSize=args.batch_size,
        numLayers=args.num_layers,
        dictLen=len(embwZH),
        embwEN=embwEN,
        embwZH=embwZH,
        device=device,
        start_TF_rate=args.tf_rate
    ).to(device)
    
    dataset1 = DatasetBase(  # 数据准备
        tokens_list_en=trainEN,
        tokens_list_zh=trainZH
    )
    dataset2 = DatasetBase(
        tokens_list_en=testEN,
        tokens_list_zh=testZH
    )
    trainLoader = DataLoader(
        dataset=dataset1,
        shuffle=True,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )
    evalLoader = DataLoader(
        dataset=dataset2,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=evalcollate_fn
    )
    load_config(
        model=model,
        target_path="base",
        para_name=args.load_para,
        if_load_or_not=args.if_load
    )
    if not args.optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fun = nn.CrossEntropyLoss()   # NER状态下只能使用交叉熵
    for epoch in range(args.epochs):
        iteration = tqdm(trainLoader, desc="Running training..")

        model.train()
        for step, batch in enumerate(iteration):
            out = model(batch[0], batch[1])
            y = torch.tensor(batch[1], dtype=torch.long).to(device)
            acc = acc_metrics(out.argmax(2), y)
            out = out.permute(0, 2, 1)
            loss = loss_fun(out, y) / args.batch_size
            wandb.log({"Train Acc:": acc})
            wandb.log({"Loss:": loss.item()})
            iteration.set_postfix(loss='{:.6f}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        iteration = tqdm(evalLoader, desc="Running eval...")
        for step, batch in enumerate(iteration):
            y = torch.tensor(batch[1], dtype=torch.long).to(device)
            out = model.evaluation(batch[0])
            acc = acc_metrics(out, y)
            wandb.log({"Eval Acc:": acc})

    if args.if_save is True:
        torch.save(model.state_dict(), "./check_points/base/" + args.save_name)