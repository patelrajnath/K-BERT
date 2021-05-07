#!/usr/bin/env bash
prefix=kaggle_lstm_crf_freeze_encoder_finetune_from_kaggle_run2

CUDA_VISIBLE_DEVICES=1 nohup python -u run_kg_luke_ner_lstm_crf.py \
	--train_path data/combined_3/train_combined_3.csv \
	--dev_path data/combined_3/dev_combined_3.csv \
	--test_path data/combined_3/test_combined_3.csv \
	--pretrained_model_path /RAJ/luke_base_500k_pretrain_kaggle_lstm_crf.tar.gz \
	--output_model_path ./outputs/kbert_${prefix}.bin \
	--output_encoder /RAJ/luke_base_models \
	--suffix_file_encoder ${prefix} \
	--kg_name /RAJ/ent_vocab_custom_cleaned_freq-sorted \
	--classifier lstm_crf \
	--batch_size 16 \
	--report_steps 100 \
	--epochs 30 \
	--seed 100 \
	--output_file_prefix outputs/evaluation/${prefix} \
	--eval_range_with_types \
	--log_file ./outputs/logs/${prefix}.log \
	> ./outputs/logs/kbert_stdout_${prefix}.log
	#--freeze_encoder_weights \
	#--reverse_order \
	#--use_subword_tag \
