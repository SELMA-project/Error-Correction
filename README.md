# Error Correction
This repository contains prototypical work on error correction. The first approach is called **ConstDecoder** which doesn't tackle specific categories of errors but rather operates on all types of errors (e.g. grammatical or entity related errors.). On the other hand, **SpellMapper** allows for user input by specifying a list of words. The goal is to correct misspelled entities.

## ConstDecoder
[Original ConstDecoder repo](https://github.com/yangjingyuan/ConstDecoder/tree/main)

[Paper](https://arxiv.org/pdf/2208.04641.pdf)
### Setup
```
conda create -n constdecoder python==3.10.12
conda activate constdecoder
pip install -r requirements.txt
```
### Data Preparation
In order to be able to train / test the performance of the ConstDecoder, you need to create your datasets in the following format:

```json
{
    "1":{
        "RAW":"ground truth sentence",
        "ASR":"asr hypothesis",
    },
    "2":{
        "RAW":"ground truth sentence",
        "ASR":"asr hypothesis",
    },
    ...
}
```

These files should live in e.g. `/ConstDecoder/datasets/YOUR_DATASET/{train,test,valid}.json`. Examplary files can be found under `/ConstDecoder/datasets/CommonVoice` (German).

### Train
If you want to train your own ConstDecoder model, define `PRETRAINED_MODEL` in `run_train_own.sh` which is the pre-trained BERT model that you want to use for encoding and tokenization as well as `YOUR_DATASET` which is the name of the dataset that you created during dataset preparation in the previous step. Depending on which language you choose to work on, you might want to set the `STRIP_ACCENTS` flag accordingly. The same also applies to the configuration parameters of the BERT model, like for example `vocab_size` or `max_src_len`.

```
# Setting PRETRAINED_MODEL=dbmdz/bert-base-german-uncased and YOUR_DATASET=CommonVoice enables you to execute an examplary training run.

cd ConstDecoder/models/ConstDecoder/scripts/

sh run_train_own.sh
```

Running this bash file yields two different things.

1. the stored models under `./models.${YOUR_DATASET}/`. If you only want to keep the best model instead of a model for every epoch after the training finisehd, uncomment the last two lines of the bash script.
2. a log of the training `train_${YOUR_DATASET}.log` which contains info on losses and WERs.

### Test

```
# Same as before, setting PRETRAINED_MODEL=dbmdz/bert-base-german-uncased and YOUR_DATASET=CommonVoice enables you to execute an examplary test run.

cd ConstDecoder/models/ConstDecoder/scripts/

sh run_eval_own.sh
```

Running this bash file yields one file, `eval_${YOUR_DATASET}_test.log`, that contains information about individual groundtruths (GOLD), asr hypotheses (ASR), the output of the ConstDecoder (Corrected), as well as the WERs before and after applying the ConstDecoder to the test set.


### Inference
For inference, please use the script `ConstDecoder/models/ConstDecoder/constdecoder_inference.py` as follows:

```
cd ConstDecoder/models/ConstDecoder/

python constdecoder_inference.py -sentence "A sentence containing errors" -trained_model scripts/models.YOUR_DATASET/best.pt -pretrained_bert NAME_PRETRAINED_BERT_TOKENIZER -vocab BERT_VOCAB_SIZE -max_src_len MAX_NUM_TOKENS_BERT -hidden_size NUM_HIDDEN_UNITS_BERT -strip_accents BOOL_STRIP_ACCENTS
```
