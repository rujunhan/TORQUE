#!/bin/bash

task="torque"
model="roberta-large"
suffix="_end2end_final.json"
model_dir=""

python eval_end_to_end.py --task_name ${task} --do_lower_case --model ${model} --file_suffix ${suffix} --data_dir ../data/ --model_dir output/${model_dir}/  --max_seq_length 178 --eval_batch_size 12
