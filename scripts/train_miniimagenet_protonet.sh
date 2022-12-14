CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
		    --exp_name meta_miniimagenet_protonet \
		    --data.src MiniImageNet \
		    --data.n_datasets_train 800 \
		    --model.base ProtoNet \
		    --train.lr 0.001 \
		    --train.lr_decay_epoch 40 \
		    --train.optimizer Adam
