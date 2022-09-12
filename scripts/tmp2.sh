NDATASETSCAL=500
NSHOTSCAL=500
NSHOTSCALTEST=20
NSHOTSTEST=5
NWAYS=5
EPS=0.1
DELTA=0.1
ALPHA=1e-5
PRETRAINED=snapshots/meta_miniimagenet_protonet/model_params_best
DATASET=MiniImageNet

for i in {1..100}
do
    CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
			--exp_name exp_${DATASET}_protonet_pac_meta_ps_n_datasets_cal_${NDATASETSCAL}_n_ways_${NWAYS}_n_shots_cal_${NSHOTSCAL}_eps_${EPS}_delta_${DELTA}_alpha_${ALPHA}_expid_${i} \
			--data.src $DATASET							\
			--model.base ProtoNet							\
			--model.path_pretrained $PRETRAINED					\
			--train_ps.method pac_meta_predset_CP					\
			--data.n_datasets_cal $NDATASETSCAL					\
			--data.n_ways $NWAYS							\
			--data.n_shots_cal $NSHOTSCAL						\
			--train_ps.eps $EPS							\
			--train_ps.delta $DELTA							\
			--train_ps.alpha $ALPHA
done
