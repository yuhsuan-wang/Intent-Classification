#Homework 1 ADL NTU 109 Spring

## Download models and data
```shell
# data will be downloaded as cache file, models will be downloaded as ckpt/slot/best.pt and ckpt/intent/best.pt
bash download.sh
```

## Intent Classification Training
```shell
python train_intent.py
```

## Intent Classification Prediction
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

