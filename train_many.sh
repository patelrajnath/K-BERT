#!/bin/bash

for kg in /RAJ/ent_vocab_custom_filtered_sorted.json /RAJ/ent_vocab_custom_cleaned_freq-sorted;
do
	for n in 2 5 10 15 20 30 50;
	do
		echo $kg
		echo $n
		prefix=max-ent-${n}
		kg_filename="$(basename -- $kg)"
		kg_filename="${kg_filename%%.*}"
		echo $kg_filename
		CUDA_VISIBLE_DEVICES=0 nohup python -u run_kg_luke_ner.py \
			--train_path data/conll_2003/eng.train.train.csv \
			--dev_path data/conll_2003/eng.testb.dev.csv \
			--test_path data/conll_2003/eng.testa.dev.csv \
			--pretrained_model_path /RAJ/luke_base_500k.tar.gz \
			--output_model_path ./outputs/kbert_conll2003_${prefix}_${kg_filename}.bin \
			--kg_name $kg \
			--batch_size 16 \
			--report_steps 100 \
			--epochs 30 \
			--use_subword_tag \
			--use_kg \
			--max_entities ${n} \
			--eval_range_with_types \
			--output_file_prefix outputs/evaluation/${prefix}_${kg_filename} \
			> ./outputs/logs/kbert_conll_${prefix}_${kg_filename}.log
			#--reverse_order \
			#--voting_choicer \
	done
done
