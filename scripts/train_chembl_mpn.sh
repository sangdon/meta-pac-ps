CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
		    --exp_name meta_chembl_mpn \
		    --data.src Chembl \
		    --data.n_datasets_train 800 \
		    --model.base ChemblMPN \
		    --train.lr 0.001 \
		    --train.lr_decay_epoch 20 \
		    --train.optimizer Adam
