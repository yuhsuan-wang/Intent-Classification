# Intent Classification
Use LSTM to classify texts' intent

## Download models and data
```shell
# models will be downloaded as ckpt/intent/best.pt
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

