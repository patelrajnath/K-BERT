#!/usr/bin/env bash
prefix=max-ent-10_filtered_save_encoder
CUDA_VISIBLE_DEVICES=0 nohup python -u run_kg_luke_ner.py \
    --train_path data/conll_2003/eng.train.train_filtered.csv \
    --dev_path data/conll_2003/eng.testb.dev_filtered.csv \
    --test_path data/conll_2003/eng.testa.dev_filtered.csv \
    --kg_name /RAJ/ent_vocab_custom_cleaned_freq-sorted \
    --pretrained_model_path /RAJ/luke_base_500k.tar.gz \
    --output_model_path ./outputs/kbert_conll2003_${prefix}.bin \
    --output_encoder /RAJ/luke_base_500k_kg_pretrain \
    --batch_size 32 \
    --report_steps 100 \
    --epochs 30 \
    --use_subword_tag \
    --output_file_prefix outputs/${prefix} \
    --eval_range_with_types \
    --use_kg \
    --max_entities 10 \
    > ./outputs/logs/kbert_conll_${prefix}.log
    #--reverse_order \
