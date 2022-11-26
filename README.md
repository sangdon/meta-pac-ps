# PAC Prediction Sets for Meta-Learning
This repository is the PyTorch implementation of [PAC Prediction Sets for Meta-Learning](https://arxiv.org/abs/2207.02440) (NeurIPS22).
This code generates a prediction set that satisfies the probably approximately correct (PAC) guarantee for meta learning. 

## Mini-ImageNet

## FewRel

## CDC Heart

Download the Heart dataset as follows:
```
cd data/heart
././download.sh
```

Train a Prototypical network as follows:
```
./scripts/train_heart_protonet.sh
```

Construct a meta PAC prediction set along with baselines as follows:
```
./scripts/cal_heart_protonet.sh
```

