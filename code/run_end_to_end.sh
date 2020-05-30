#!/bin/bash
task="torque"
batchsizes=(6)
ratio=1.0
epoch=10
mlp_hid_size=64
model="roberta-large"
ga=2
prefix="end_to_end_model"
suffix="_end2end_final.json"
for s in "${batchsizes[@]}"
do
    learningrates=(1e-5 2e-5)

    for l in "${learningrates[@]}"
    do
        seeds=( 7 24 123 )
        for seed in "${seeds[@]}"
        do
            python run_end_to_end.py --task_name "${task}" --do_train --do_eval --do_lower_case --mlp_hid_size ${mlp_hid_size} --model ${model} --data_dir ../data/ --file_suffix ${suffix} --train_ratio ${ratio} --max_seq_length 178 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e}  --gradient_accumulation_steps=${ga}  --output_dir output/${prefix}_${model}_batch_${s}_lr_${l}_epochs_${epoch}_seed_${seed}_${ratio} --seed ${seed}
        done
    done
done
