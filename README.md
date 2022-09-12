# Meta PAC Prediction Sets

## Heart

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

