GPUID=2
NDATASETSCAL=500
NSHOTSCAL=500
NSHOTSCALTEST=20
NSHOTSTEST=100
NWAYS=5
EPS=0.1
DELTA=0.1 #TODO: switch alpha and delta
ALPHA=1e-5
PRETRAINED=snapshots/meta_fewrel_protonet/model_params_best
DATASET=FewRel
MODEL=ProtoNetNLP

for i in {1..100}
do
    CUDA_VISIBLE_DEVICES=$GPUID python3 main_cls_fewshot.py \
			--exp_name exp_${DATASET}_protonet_meta_ps_ideal_n_datasets_cal_${NDATASETSCAL}_n_ways_${NWAYS}_n_shots_cal_${NSHOTSCALTEST}_eps_${EPS}_delta_${ALPHA}_expid_${i} \
			--data.src $DATASET \
			--model.base $MODEL \
			--model.path_pretrained $PRETRAINED \
			--train_ps.method meta_ps_ideal \
			--data.n_ways $NWAYS \
			--data.n_shots_cal $NSHOTSCALTEST \
			--data.n_shots_test $((NSHOTSCALTEST+NSHOTSTEST)) \
			--train_ps.eps $EPS \
			--train_ps.delta $ALPHA

done
