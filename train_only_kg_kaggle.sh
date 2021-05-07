#!/usr/bin/env bash
prefix=kaggle_pretrain_lstm_crf

CUDA_VISIBLE_DEVICES=1 nohup python -u run_kg_luke_ner_lstm_crf.py \
	--train_path data/combined_3/train_combined_3_normalized.csv \
	--dev_path data/combined_3/dev_combined_3_normalized.csv \
	--test_path data/combined_3/test_combined_3_normalized.csv \
	--pretrained_model_path /RAJ/luke_base_500k.tar.gz \
	--output_model_path ./outputs/kbert_${prefix}.bin \
	--output_encoder /RAJ/luke_base_kg_pretrain_models \
	--suffix_file_encoder ${prefix} \
	--kg_name /RAJ/ent_vocab_custom_cleaned_freq-sorted \
	--use_kg \
	--max_entities 15 \
	--classifier lstm_crf \
	--batch_size 16 \
	--report_steps 100 \
	--epochs 30 \
	--output_file_prefix outputs/evaluation/${prefix} \
	--eval_range_with_types \
	--log_file ./outputs/logs/${prefix}.log \
	> ./outputs/logs/kbert_stdout_${prefix}.log
	#--reverse_order \
	#--use_subword_tag \
