Introduction
---------
This is the reproduction package for  **TORQUE: A Reading Comprehension Dataset of Temporal Ordering Questions** in EMNLP'2020 [(link)](https://allennlp.org/torque.html).

Download raw data
----------
Please see [here](https://github.com/rujunhan/TORQUE/tree/master/raw).

Preprocessing
----------
You may want to run the following python script first if `averaged_perceptron_tagger` isn't already installed on your computer.
```
import nltk
nltk.download('averaged_perceptron_tagger')
```

And then, to process TORQUE for the modeling, run from the root dir of this project,
```
mkdir -p data
python code/process_train.py
python code/process_dev.py
python code/process_test.py
```

You may find some output message like below, which is expected.
```
...
case 2: single character ('(12,13)', 's')
ignore case: WDT
...
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


