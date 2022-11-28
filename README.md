# PAC Prediction Sets for Meta-Learning
This repository is the PyTorch implementation of [PAC Prediction Sets for Meta-Learning](https://arxiv.org/abs/2207.02440) (NeurIPS22).
This code generates a prediction set that satisfies the probably approximately correct (PAC) guarantee for meta learning. 

<p align="center"><img src=".github/teaser.png" width="500">

## Mini-ImageNet Dataset

Download the Mini-ImageNet dataset from the [original repo](https://github.com/yaoyao-liu/mini-imagenet-tools). 
In particular, download the postprocessed dataset from [this link](https://drive.google.com/open?id=137M9jEv8nw0agovbUiEN_fPl_waJ2jIj), and
put it under `data/miniimagenet` (i.e., 'data/miniimagenet/mini-imagenet'). 
The following script takes care of the rest postprocessing. 
```
cd data/miniimagenet
./process.sh
```

Train a Prototypical network as follows:
```  
./scripts/train_miniimagenet_protonet.sh 
```
  
Construct and evaluate a meta PAC prediction set along with baselines as follows:
```
./scripts/cal_miniimagenet_protonet.sh
```
  
To reproduce evaluation results, run the following script to generate plots:
```
./scripts/plot_miniimagenet.sh
```

## FewRel Dataset
  
We use [FewRel 1.0 and a related toolkit](https://github.com/thunlp/FewRel). The required part of the toolkit and dataset are included in this repository.
the following script initializes the toolkit and the dataset for you.
```
cd data/fewrel
./process.sh
```
Train a Prototypical network as follows:
```  
./scripts/train_fewrel_protonet.sh 
```
  
Construct and evaluate a meta PAC prediction set along with baselines as follows:
```
./scripts/cal_fewrel_protonet.sh
```
  
To reproduce evaluation results, run the following script to generate plots:
```
./scripts/plot_fewrel.sh
```


## CDC Heart Dataset

Download the Heart dataset as follows:
```
cd data/heart
./download.sh
```

Train a Prototypical network as follows:
```
./scripts/train_heart_protonet.sh
```

Construct and evaluate a meta PAC prediction set along with baselines as follows:
```
./scripts/cal_heart_protonet.sh
```

To reproduce evaluation results, run the following script to generate plots:
```
./scripts/plot_heart.sh
```

