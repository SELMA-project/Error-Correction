#!/bin/bash

: ' How to test uncased ConstDecoder:

Depending on the language you want to train ConstDecoder on, select the desired BERT 
model. (e.g. bert-base-multilingual, or dbmdz/bert-base-german-uncased). Based on that 
decision, make sure to set STRIP_ACCENTS accordingly as it directly impacts the
performance of the model. For accented language like e.g. French, or German you want to 
set it to False.

To test ConstDecoder on your own data, generate test.json 
(e.g. ../../../datasets/${YOUR_DATASET}/test.json) in the following format:

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

where "RAW" is the ground truth and "ASR" is the hypothesis of e.g. Whisper.
Adjust the name of the folder to YOUR_DATASET. 

Make sure to configure the flags below according to the config file of the 
PRETRAINED_MODEL you choose to use.

cd models/ConstDecoder/scripts/
sh run_eval_own.sh
'

PRETRAINED_MODEL=YOUR_CHOICE_OF_PRETRAINED_BERT_MODEL
STRIP_ACCENTS=false
YOUR_DATASET=NAME_OF_DATASET

nohup python -u ../eval.py \
    --base_model $PRETRAINED_MODEL \
    --tokenizer_name $PRETRAINED_MODEL \
    --strip_accents $STRIP_ACCENTS \
    --model_path ./models.${YOUR_DATASET}/best.pt \
    --test_data_path ../../../datasets/${YOUR_DATASET}/test.json \
    --device 0 \
    --tag_pdrop 0.2 \
    --decoder_proj_pdrop 0.2 \
    --tag_hidden_size 768 \
    --tag_size 3 \
    --alpha 3.0 \
    --change_weight 1.5 \
    --max_src_len 512 \
    --vocab_size 31102 \
    --pad_token_id 0 \
    --max_add_len 10 >eval_${YOUR_DATASET}_test.log 2>&1 &