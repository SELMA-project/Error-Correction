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


## SpellMapper
[Original SpellMapper repo](https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/spellchecking_asr_customization) 

[Paper](https://arxiv.org/pdf/2306.02317.pdf)

### Setup 

```shell
conda create --name spellmapper python==3.10.12
conda activate spellmapper
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
pip install Cython
python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
```

### Data Preparation

As described in the [author's repo](https://github.com/bene-ges/nemo_compatible/tree/main/scripts/nlp/en_spellmapper), multiple steps need to be followed. The following description of steps is based on `English`.

1. Have an extensive list of entities (`entities.uniq`) at hand. Originally, [YagoTypes](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/) was used for this (~5 Million entities). 

```
# inside entities.uniq, e.g.
Aaron_Judge
Obednik
...
```
If you have this file, run 

```
cd SpellMapper/nemo_compatible
sh dataset_preparation.sh
```


**If** you don't already have a (faulty) transcription of your list of entities in the following format `pred_ctc.all.json`:

```json
{"audio_filepath": "tts/0.wav", "text": "cycling", "pred_text": "cycling"}
{"audio_filepath": "tts/1.wav", "text": "brussels", "pred_text": "where says"}
{"audio_filepath": "tts/3.wav", "text": "thailand", "pred_text": "thailand"}
{"audio_filepath": "tts/4.wav", "text": "hartley", "pred_text": "harkley"}
...
```

- Feed entities to G2P to get phonemes by using `dataset_preparation/run_g2p.sh`
- Feed phonetic inputs to TTS, generate wav files. Feed wav files to ASR, get ASR hypotheses (also misspelled) using `dataset_preparation/run_tts_and_asr.sh`

**Else**:

2. Align parallel corpus of "correct + misspelled" phrases with Giza++, count frequencies of ngram mappings using `dataset_preparation/get_ngram_mappings.sh`. **NOTE** make sure to install GIZA++ as described in `dataset_preparation/get_ngram_mappings.sh`
3. Generate training data using `dataset_preparation/build_training_data.sh` Generate `config.json`, `label_map.txt`, `semiotic_classes.txt` using `dataset_preparation/generate_configs.sh`. Make sure to specify the configuration details of the BERT model you want to use. [Optional] Convert training dataset to tarred files using `convert_dataset_to_tarred.sh`


### Train
For the next step you need to clone the Nemo repository if you haven't already done so for the usage of `dataset_preparation/run_g2p.sh`

```
git clone https://github.com/NVIDIA/NeMo NeMo
```

Train SpellMapper (part of NeMo repository) using `[NeMo]/examples/nlp/spellchecking_asr_customization/run_training.sh` or `[NeMo]/examples/nlp/spellchecking_asr_customization/run_training_tarred.sh`

Inside `run_training.sh`, adjust the following flags according to you choice of language:

```
lang="en" \
model.language_model.pretrained_model_name=PRETRAINED_BERT \
```

and set `DATA_PATH` to the directory where you saved `test.tsv` and `train.tsv` to when you ran `dataset_preparation/build_training_data.sh`.

### Test 
By default the testing scripts use a pretrained checkpoint (here English) to reproduce results of the paper e.g. `evaluation/test_on_userlibri.sh`. 
When using your own data you want to create `${WORKDIR}/vocabs` which contains a .txt file with your custom vocabulary and `${WORKDIR}/manifests` that contains a file similiar to the previously mentioned `pred_ctc.all.json` called `manifest_ctc.json` which looks as follows.

```json
{"text": "ground truth sentence", "audio_filepath": "/nemo_compatible/scripts/nlp/en_spellmapper/evaluation/corpus/audio_data/test-clean/speaker-1089-book-4217/1089-134686-0000.wav", "doc_id": "4217", "pred_text": "asr hypothesis"}
{"text": "ground truth sentence", "audio_filepath": "/nemo_compatible/scripts/nlp/en_spellmapper/evaluation/corpus/audio_data/test-clean/speaker-1089-book-4217/1089-134686-0001.wav", "doc_id": "4217", "pred_text": "asr hypothesis"}
...
```


### Inference
You can use a pre-trained SpellMapper model by running `[NeMo]/examples/nlp/spellchecking_asr_customization/run_infer.sh`. Remember to set the paths of 

```
PRETRAINED_MODEL
NGRAM_MAPPINGS
BIG_SAMPLE
INPUT_MANIFEST
CUSTOM_VOCAB
```
to your files and the `lang=en` flag to your desired language when encoutering the `# Run SpellMapper inference` section.
