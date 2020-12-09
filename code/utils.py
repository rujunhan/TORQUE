from typing import Iterator, List, Mapping, Union, Optional, Set
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import json
import pickle
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir, split, suffix):
    filename = "%s%s%s" % (data_dir, split, suffix)
    print("==========load data from %s ===========" % filename)
    with open(filename, "r") as read_file:
        return json.load(read_file)

def select_field(data, field):
    # collect a list of field in data                                                                  
    # fields: 'label', 'offset', 'input_ids, 'mask_ids', 'segment_ids', 'question_id'                            
    return [ex[field] for ex in data]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def exact_match(question_ids, labels, predictions):
    em = defaultdict(list)
    for q, l, p in zip(question_ids, labels, predictions):
        em[q].append(l == p)
    print("Total %s questions" % len(em))
    return float(sum([all(v) for v in em.values()])) / float(len(em))

def sample_errors(passages, questions, answers, labels, preds, label_class="Positive", num=50):
    assert len(passages) == len(preds)
    assert len(questions) == len(preds)
    assert len(answers) == len(preds)
    errors = []
    outfile = open("%s_error_samples.tsv" % label_class, 'w')
    outfile.write("Passage\tQuestion\tAnswer-span\tAnswer-offset\tAnswer-label\tAnswer-prediction\n")
    count = 0
    for pa, q, a, l, p in zip(passages, questions, answers, labels, preds):
        if count >= num:
            continue
        if l == label_class and l != p:
            outfile.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (" ".join(pa), q, a['span'], a['idx'], l, p))
            count += 1
    outfile.close()
    return

def get_train_dev_ids(data_dir, data_type):
    trainIds = [f.strip() for f in open("%s/%s/trainIds.txt" % (data_dir, data_type))]
    devIds = [f.strip() for f in open("%s/%s/devIds.txt" % (data_dir, data_type))]
    return trainIds, devIds


def convert_to_features_roberta_no_label(data, tokenizer, max_length=150, evaluation=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0  # to show global max_len without truncating

    for k, v in data.items():
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])

        new_tokens = ["</s>", "</s>"]  # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1  # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)

        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>

        # duplicate P + Q for each answer
        offsets = []
        for kk, vv in enumerate(v['context']):
            offsets.append(orig_to_tok_map[kk] + len(question) + 1)

        sample = {'offset': offsets,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'question_id': k}
        # add these three field for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
            sample['question'] = v['question']
            sample['question_cluster'] = v['question_cluster']
            sample['cluster_size'] = v['cluster_size']
        samples.append(sample)

        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

def convert_to_features_roberta(data, tokenizer, max_length=150, evaluation=False,
                                instance=True, end_to_end=False):
    # each sample will have <s> Question </s> </s> Context </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 

    for k, v in data.items():
        segment_ids = []
        start_token = ['<s>']
        question = tokenizer.tokenize(v['question'])

        new_tokens = ["</s>", "</s>"] # two sent sep symbols according the huggingface documentation
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            
        new_tokens.append("</s>")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending </s>

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            
        segment_ids = [0] * len(tokenized_ids)
        # mask ids                                                                                            
        mask_ids = [1] * len(tokenized_ids)

         # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample
        # offset: </s> </s> counted in orig_to_tok_map already, so only need to worry about <s>
        if end_to_end:
            # duplicate P + Q for each answer
            labels, offsets = [], []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}
            # add these three field for qualitative analysis                                            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            samples.append(sample)
        else:
            # no duplicate P + Q
            labels, offsets = [], []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}
            
            # add these three field for qualitative analysis         
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
                
            samples.append(sample)
            
        # check some example data
        if counter < 0:
            print(sample)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

def flatten_answers(answers):    
    # flatten answers and use batch length to map back to the original input   
    offsets = [a for ans in answers for a in ans[1]]
    labels = [a for ans in answers for a in ans[0]]
    lengths = [len(ans[0]) for ans in answers]

    assert len(offsets)  == sum(lengths)
    assert len(labels) == sum(lengths)
    
    return offsets, labels, lengths

def flatten_answers_no_label(answers):
    offsets = [a for ans in answers for a in ans]
    lengths = [len(ans) for ans in answers]

    assert len(offsets) == sum(lengths)
    return offsets, lengths


def convert_to_features(data, tokenizer, max_length=150, evaluation=False,
                        instance=True, end_to_end=False):
    # each sample will have [CLS] + Question + [SEP] + Context                                                                                 
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating                                                                             
    for k, v in data.items():
        segment_ids = []
        start_token = ['[CLS]']
        question = tokenizer.tokenize(v['question'])

        # the following bert tokenized context starts / end with ['SEP']                                                                       
        new_tokens = ["[SEP]"]
        orig_to_tok_map = []
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)

        new_tokens.append("[SEP]")
        length = len(new_tokens)
        orig_to_tok_map.append(length)
        assert len(orig_to_tok_map) == len(v['context']) + 1 # account for ending ['SEP']                                                      

        # following the bert convention for calculating segment ids                                                                            
        segment_ids = [0] * (len(question) + 2) + [1] * (len(new_tokens) - 1)

        # mask ids                                                                                                                             
        mask_ids = [1] * len(segment_ids)

        tokenized_ids = tokenizer.convert_tokens_to_ids(start_token + question + new_tokens)
        assert len(tokenized_ids) == len(segment_ids)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        # truncate long sequence, but we can simply set max_length > global_max                                                                
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            assert len(tokenized_ids) == max_length

        # padding                                                                                                                              
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.                                                                                              
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding

        if end_to_end:
            labels, offsets = [], []
            for kk, vv in enumerate(v['answers']['labels']):
                labels.append(vv)
                offsets.append(orig_to_tok_map[kk] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}
            # add these three field for qualitative analysis                                                                
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['question_cluster'] = v['question_cluster']
                sample['cluster_size'] = v['cluster_size']
                sample['answer'] = v['answers']
                sample['individual_answers'] = [a['labels'] for a in v['individual_answers']]
            samples.append(sample)
        else:
            # no duplicate P + Q                                                  
            labels, offsets = [], []
            for vv in v['answers'].values():
                labels.append(vv['label'])
                offsets.append(orig_to_tok_map[vv['idx']] + len(question) + 1)

            sample = {'label': labels,
                      'offset': offsets,
                      'input_ids': tokenized_ids,
                      'mask_ids': mask_ids,
                      'segment_ids': segment_ids,
                      'question_id': k}

            # add these three field for qualitative analysis            
            if evaluation:
                sample['passage'] = v['context']
                sample['question'] = v['question']
                sample['answer'] = v['answers']
                individual_answers = []
                for vv in v['individual_answers']:
                    individual_answers.append([a['label'] for a in vv.values()])
                sample['individual_answers'] = individual_answers
            samples.append(sample)
            
        # check some example data                                                                                                              
        if counter < 0:
            print(k)
            print(v)
            print(tokenized_ids)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

def cal_f1(pred_labels, true_labels, label_map, log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

    num_tests = len(true_labels)

    total_true = Counter(true_labels)
    total_pred = Counter(pred_labels)

    labels = list(label_map)

    n_correct = 0
    n_true = 0
    n_pred = 0

    label_map
    # we only need positive f1 score
    exclude_labels = ['Negative']
    for label in labels:
        if label not in exclude_labels:
            true_count = total_true.get(label, 0)
            pred_count = total_pred.get(label, 0)

            n_true += true_count
            n_pred += pred_count

            correct_count = len([l for l in range(len(pred_labels))
                                 if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
            n_correct += correct_count
    if log:
        logger.info("Correct: %d\tTrue: %d\tPred: %d" % (n_correct, n_true, n_pred))
    precision = safe_division(n_correct, n_pred)
    recall = safe_division(n_correct, n_true)
    f1_score = safe_division(2.0 * precision * recall, precision + recall)
    if log:
        logger.info("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))
    return f1_score
