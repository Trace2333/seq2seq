from utils.args import ArgsParse
from trainers.train_base import trainer_base
from trainers.train_no_embw import trainer_base_no_embw
from utils.sh_create import sh_create, json_create


args = ArgsParse()

# 模型超参数

args.batch_size = 32
args.lr = 1e-3
args.epochs = 10
args.optimizer = "Adam"
args.lossfun = "CrossEntropyLoss"
args.model_name = "seq2seq_base"
args.device = 'cuda'
args.evaluation_epochs = 1
args.input_size = 300
args.hidden_size = 300
args.num_layers = 1
args.tf_rate = 0.6

# 文件操作
args.train_file = "train.py"
args.if_load = False
args.if_save = True
args.load_para = ""
args.save_name = "epoch=10_tf=1_lr=1e-3.pth"


# wandb设置
args.project = "seq2seq-LSTM-base"
args.notes = "epoch=10,lr=1e-3,tf=1"

sh_create("run.sh", json_create(args))
#trainer_base(args)
trainer_base_no_embw(args)
