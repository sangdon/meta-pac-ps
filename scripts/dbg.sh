NDATASETSCAL=500
NSHOTSCAL=500
NSHOTSCALTEST=20
NSHOTSTEST=100
NWAYS=5
EPS=0.1
DELTA=0.1 #TODO: switch alpha and delta
ALPHA=1e-5
PRETRAINED=snapshots/meta_miniimagenet_protonet/model_params_best
DATASET=MiniImageNet

for i in {1..100}
do
    CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
			--exp_name dbg123 \
			--data.src $DATASET \
			--model.base ProtoNet \
			--model.path_pretrained $PRETRAINED \
			--train_ps.method meta_pac_ps \
			--data.n_datasets_cal $NDATASETSCAL \
			--data.n_ways $NWAYS \
			--data.n_shots_cal $NSHOTSCAL \
			--train_ps.eps $EPS \
			--train_ps.delta $DELTA \
			--train_ps.alpha $ALPHA

done
