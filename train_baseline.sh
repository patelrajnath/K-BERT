#!/usr/bin/env bash
prefix=conll_baseline_lstm_crf

CUDA_VISIBLE_DEVICES=0 nohup python -u run_kg_luke_ner_lstm_crf.py \
     --train_path data/conll_2003/eng.train.train.csv \
     --dev_path data/conll_2003/eng.testb.dev.csv \
     --test_path data/conll_2003/eng.testa.dev.csv \
     --pretrained_model_path /RAJ/luke_base_500k.tar.gz \
     --output_model_path ./outputs/kbert_${prefix}.bin \
     --output_encoder /RAJ/luke_base_500k_kg_pretrain \
     --kg_name /RAJ/ent_vocab_custom_cleaned_freq-sorted \
     --batch_size 16 \
     --report_steps 100 \
     --epochs 30 \
     --classifier lstm_crf \
     --output_file_prefix outputs/evaluation/${prefix} \
     --eval_range_with_types \
     --log_file ./outputs/logs/${prefix}.log \
     > ./outputs/logs/kbert_stdout_${prefix}.log
    # --use_kg \
    #--reverse_order \
    #--use_subword_tag \
