#!/usr/bin/env bash
prefix=test_avaluation_code_lstm_crf

CUDA_VISIBLE_DEVICES=1 nohup python -u run_kg_luke_ner_lstm_crf.py \
     --train_path data/conll_2003/eng.train.train_normalized.csv \
     --dev_path data/conll_2003/eng.testb.dev_normalized.csv \
     --test_path data/conll_2003/eng.testa.dev_normalized.csv \
     --pretrained_model_path /RAJ/luke_base_500k.tar.gz \
     --output_model_path ./outputs/kbert_${prefix}.bin \
     --output_encoder /RAJ/luke_base_500k_kg_pretrain \
     --kg_name /RAJ/ent_vocab_custom_cleaned_freq-sorted \
     --use_kg \
     --max_entities 15 \
     --batch_size 8 \
     --voting_choicer \
     --report_steps 100 \
     --epochs 30 \
     --output_file_prefix outputs/evaluation/${prefix} \
     --eval_range_with_types \
     --log_file ./outputs/logs/${prefix}.log \
     > ./outputs/logs/kbert_stdout_${prefix}.log
    #--reverse_order \
    #--use_subword_tag \
