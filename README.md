Introduction
---------
This is the reproduction package for the submission entitled **TORQUE: A Reading Comprehension Dataset of Temporal Ordering Questions** to EMNLP'2020. Also comes along with it is the entire dataset of TORQUE already split into train/dev/test. We'll incorporate any suggestions from the reviewers and make the dataset and package public for future investigations after acceptance. Thanks in advance for reviewers' time and effort in improving this paper.


Usage
----------
0. TORQUE can be found in `data/`

1. Fine-tune pretrained LMs on TORQUE

> bash code/run_end_to_end.sh

2. Evaluate trained model and report performances

> bash code/eval_end_to_end.sh
