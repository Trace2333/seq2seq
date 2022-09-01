from utils.args import ArgsParse
from trainers.train_base import trainer_base
from utils.sh_create import sh_create, json_create


args = ArgsParse()

# 模型超参数

args.batch_size =16
args.lr = 1e-3
args.epochs = 1
args.evaluation_epochs = 1
args.optimizer = "Adam"
args.lossfun = "CrossEntropyLoss"
args.model_name = "seq2seq_base"
args.device = 'cuda'

args.input_size = 300
args.hidden_size = 300
args.num_layers = 1

# 文件操作

args.train_file = "train.py"
args.if_load = False
args.if_save = False
args.load_para = ""
args.save_name = ""


# wandb设置
args.project = "seq2seq-LSTM-base"
args.notes = "Fro test in the rewrite progress"

sh_create("run.sh", json_create(args))
trainer_base(args)
