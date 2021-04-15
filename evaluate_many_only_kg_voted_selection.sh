#!/bin/bash

for kg in /RAJ/ent_vocab_custom_cleaned_freq-sorted #/RAJ/ent_vocab_custom_filtered_sorted.json;
do
	for n in 15 #10 15 20 30 50;
	do
		echo $kg
		echo $n
		prefix=voting-choicer_max-ent-${n}
		kg_filename="$(basename -- $kg)"
		kg_filename="${kg_filename%%.*}"
		echo $kg_filename
		CUDA_VISIBLE_DEVICES=1 nohup python -u kg_luke_ner_evaluate.py \
		  --train_path data/conll_2003/eng.train.train_filtered.csv \
		  --dev_path data/conll_2003/eng.testb.dev.csv \
		  --test_path data/conll_2003/eng.testa.dev.csv \
			--pretrained_model_path /RAJ/luke_base_500k.tar.gz \
			--output_model_path ./outputs/kbert_conll2003_${prefix}_${kg_filename}.bin \
			--kg_name $kg \
			--batch_size 4 \
			--report_steps 100 \
			--epochs 30 \
			--voting_choicer \
			--use_kg \
			--max_entities ${n} \
			--eval_range_with_types \
			--output_file_prefix outputs/evaluation/${prefix}_${kg_filename} \
			> ./outputs/logs-voting-chicer/original_test_kbert_conll_${prefix}_${kg_filename}.log
			#--reverse_order \
			#--use_subword_tag \
	done
done
