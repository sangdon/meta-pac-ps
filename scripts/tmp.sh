NDATASETSCAL=250
NSHOTSCAL=1500 # 500
NSHOTSCALTEST=20
NSHOTSTEST=100
NWAYS=2
EPS=0.1
DELTA=0.1 #TODO: switch alpha and delta
ALPHA=1e-5
PRETRAINED=snapshots/meta_heart_protonet/model_params_best
DATASET=Heart

for i in {1..100}
do
    CUDA_VISIBLE_DEVICES=3 python3 main_cls_fewshot.py \
			--exp_name exp_${DATASET}_protonet_meta_ps_naive_n_datasets_cal_${NDATASETSCAL}_n_ways_${NWAYS}_n_shots_cal_${NSHOTSCAL}_eps_${EPS}_delta_${ALPHA}_expid_${i} \
			--data.src $DATASET \
			--model.base ProtoNetGeneral \
			--model.path_pretrained $PRETRAINED \
			--train_ps.method meta_ps_naive \
			--data.n_datasets_cal $NDATASETSCAL \
			--data.n_ways $NWAYS \
			--data.n_shots_cal $NSHOTSCAL \
			--train_ps.eps $EPS \
			--train_ps.delta $ALPHA
done
