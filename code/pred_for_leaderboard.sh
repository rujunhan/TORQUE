#!/bin/bash
tasks=("leaderboard")
epoch=10
batch=6
mlp_hid_size=64
seed=(7 24 123)
lr=(1e-5)
model="roberta-large"
suffix="_end2end_final_unlabeld.json"
for task in "${tasks[@]}"
  do
  for l in "${lr[@]}"
    do
	  for s in "${seed[@]}"
	    do
	    dir="${task}_${model}_batch_${batch}_lr_${l}_epochs${epoch}_seed_${s}_1.0"
	    python output_pred.py \
	    --task_name ${task} \
	    --do_lower_case \
	    --model ${model} \
	    --mlp_hid_size ${mlp_hid_size} \
	    --file_suffix ${suffix} \
	    --data_dir ../data/ \
	    --model_dir output/${dir}/  \
	    --max_seq_length 178 \
	    --eval_batch_size 12 \
	    --seed ${s}
	    done
    done
done
