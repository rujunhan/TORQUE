# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm, trange
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
from models import MultitaskClassifier, MultitaskClassifierRoberta
from optimization import *
from utils import *
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the trained model are saved")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="Suffix of filename")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--eval_ratio",
                        default=0.5,
                        type=float,
                        help="portion of data for evaluation")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help="random seed for initialization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    logger.info("current task is " + str(task_name))

    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")

    if 'roberta' in args.model:
        tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_-1'
        model = MultitaskClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                                           cache_dir=cache_dir, mlp_hid=args.mlp_hid_size)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_-1'
        model = MultitaskClassifier.from_pretrained(args.model, state_dict=model_state_dict,
                                                    cache_dir=cache_dir, mlp_hid=args.mlp_hid_size)
    model.to(device)
    hyper_params = args.model_dir.split('/')[1].split('_')
    for eval_file in ['test']:  # default question, non-default question, overall

        print("=" * 50 + "Evaluating %s" % eval_file + "=" * 50)
        eval_data = load_data(args.data_dir, "individual_%s" % eval_file, args.file_suffix)
        if 'roberta' in args.model:
            eval_features = convert_to_features_roberta_no_label(eval_data, tokenizer,
                                                                max_length=args.max_seq_length, evaluation=True)
        else:
            # other LM models not implemented yet
            continue

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
        eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        eval_offsets = select_field(eval_features, 'offset')
        eval_key_indices = torch.tensor(list(range(len(eval_offsets))), dtype=torch.long)

        # collect unique question ids for EM calculation
        question_ids = select_field(eval_features, 'question_id')

        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss, eval_accuracy, best_eval_f1, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0.0, 0, 0

        all_predictions = {}
        for input_ids, input_masks, segment_ids, instance_indices in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            segment_ids = segment_ids.to(device)

            offsets, lengths = flatten_answers_no_label([eval_offsets[i]
                                                         for i in instance_indices.tolist()])

            with torch.no_grad():
                logits = model(input_ids, offsets, lengths, token_type_ids=segment_ids,
                               attention_mask=input_masks)

            logits = logits.detach().cpu().numpy()

            nb_eval_examples += logits.shape[0]
            nb_eval_steps += 1

            batch_preds = np.argmax(logits, axis=1)
            bi = 0
            for l, idx in enumerate(instance_indices):
                pred = [int(batch_preds[bi + li]) for li in range(lengths[l])]
                all_predictions[question_ids[idx]] = pred
                bi += lengths[l]

        with open("./output/test_preds_%s_%s.json" % (hyper_params[-5], args.seed), 'w') as outfile:
             json.dump(all_predictions, outfile)

if __name__ == "__main__":
    main()