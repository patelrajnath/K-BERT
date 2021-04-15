#!/usr/bin/env bash
kg=/RAJ/ent_vocab_custom_cleaned_freq-sorted
prefix=baseline_filtered
CUDA_VISIBLE_DEVICES=1 nohup python -u kg_luke_ner_evaluate.py \
	--train_path data/conll_2003/eng.train.train_filtered.csv \
	--dev_path data/conll_2003/eng.testb.dev.csv \
	--test_path data/conll_2003/eng.testa.dev.csv \
	--pretrained_model_path /RAJ/luke_base_500k.tar.gz \
	--output_model_path ./outputs/kbert_conll2003_${prefix}.bin \
	--kg_name /RAJ/ent_vocab_custom_cleaned_freq-sorted \
	--batch_size 4 \
	--eval_range_with_types \
	--output_file_prefix outputs/evaluation/${prefix} \
        > ./outputs/logs-voting-chicer/original_test_kbert_conll_${prefix}.log
	#--dev_path data/conll_2003/eng.testb.dev.csv \
	#--test_path data/conll_2003/eng.testa.dev.csv \
