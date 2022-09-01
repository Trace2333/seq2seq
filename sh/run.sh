python3 train.py \
	--batch_size=16 \
	--lr=0.001 \
	--epochs=1 \
	--evaluation_epochs=1 \
	--optimizer=Adam \
	--lossfun=CrossEntropyLoss \
	--model_name=seq2seq_base \
	--device=cuda \
	--input_size=300 \
	--hidden_size=300 \
	--num_layers=1 \
	--if_load=False \
	--if_save=False \
	--load_para= \
	--save_name= \
	--notes=Fro test in the rewrite progress