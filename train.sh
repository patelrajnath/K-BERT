#!/usr/bin/env bash

python run_kg_luke_ner.py \
    --train_path data/conll_2003/eng.train.train.csv \
    --dev_path data/conll_2003/eng.testb.dev.csv \
    --test_path data/conll_2003/eng.testa.dev.csv \
    --kg_name D:\\Downloads\\ent_vocab_custom --use_kg \
    --pretrained_model_path  D:\\Downloads\\luke_base_500k.tar.gz
