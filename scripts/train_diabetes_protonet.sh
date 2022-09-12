CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
		    --exp_name meta_diabetes_protonet \
		    --data.src Diabetes \
		    --data.n_datasets_train 800 \
		    --model.base ProtoNetGeneral \
		    --train.lr 0.001 \
		    --train.lr_decay_epoch 40 \
		    --train.optimizer Adam
