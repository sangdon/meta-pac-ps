CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
		    --exp_name meta_fewrel_protonet \
		    --data.src FewRel \
		    --data.n_datasets_train 800 \
		    --model.base ProtoNetNLP \
		    --train.lr 0.01 \
		    --train.lr_decay_epoch 40 \
		    --train.optimizer Adam \
		    --data.batch_size 10
