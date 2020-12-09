Introduction
---------
This is the reproduction package for the submission entitled **TORQUE: A Reading Comprehension Dataset of Temporal Ordering Questions** to EMNLP'2020. Also comes along with it is the entire dataset of TORQUE already split into train/dev/test. We'll incorporate any suggestions from the reviewers and make the dataset and package public for future investigations after acceptance. Thanks in advance for reviewers' time and effort in improving this paper.

Raw Data
----------
You can download the original TORQUE datasets from https://github.com/qiangning/TORQUE-dataset/tree/main/data/. Save all .json files there into `./raw/`.
To process TORQUE for the modeling, run,

```
python code/process_train.py
python code/process_dev.py
python code/process_test.py
```

Usage
----------
Processed TORQUE data can be found in `data/`

Fine-tune pretrained LMs on TORQUE
```
bash code/run_end_to_end.sh
```

Evaluate trained model and report performances
```
bash code/eval_end_to_end.sh
```

Output model predictions for the leaderboard
```
bash pred_for_leaderboard.sh
```
Assuming your trained model is saved in `${model_dir}/pytorch_model.bin` where `${model_dir}` corresponds to the `${output_dir}` in `run_end_to_end.sh`


