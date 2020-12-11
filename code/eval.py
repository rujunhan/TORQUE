import argparse
from utils import *

def output_gold():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=args.do_lower_case)
    data = load_data(args.data_dir, "individual_test", "_end2end_final.json")
    features = convert_to_features_roberta(data, tokenizer, max_length=178,
                                           evaluation=True, instance=False, end_to_end=True)
    labels = select_field(features, 'label')

    # collect unique question ids for EM calculation
    question_ids = select_field(features, 'question_id')

    # collect unique question culster for EM-cluster calculation
    question_cluster = select_field(features, 'question_cluster')
    question_cluster_size = select_field(features, 'cluster_size')
    idv_answers = select_field(features, 'individual_answers')

    gold = {}
    for id, l, c, cs, idv in zip(question_ids, labels, question_cluster, question_cluster_size, idv_answers):
        gold[id] = {'label': l,
                    'cluster': c,
                    'cluster_size': cs,
                    'idv_answers': idv}
        if len(gold) == 3:
            print(gold)
    with open("./output/test_gold.json", 'w') as outfile:
        json.dump(gold, outfile)
    return

label_map = {0: "Negative", 1: "Postive"}

def evaluate(golds, preds):
    all_preds, all_golds, max_f1s, macro_f1s = [], [], [], []
    f1_dist = defaultdict(list)
    em_counter = 0
    em_cluster_agg, em_cluster_relaxed, f1_cluster_80 = {}, {}, {}

    for k, v in golds.items():
        pred = preds[k]
        pred_names = [label_map[p] for p in pred]
        gold_names = [label_map[g] for g in v['label']]
        is_em = (pred_names == gold_names)

        all_preds.extend(pred)
        all_golds.extend(v['label'])

        if sum(v['label']) == 0 and sum(pred) == 0:
            macro_f1s.append(1.0)
        else:
            macro_f1s.append(cal_f1(pred_names, gold_names, {v: k for k, v in label_map.items()}))

        max_f1, instance_matched = 0, 0
        for gold in v['idv_answers']:
            label_names = [label_map[l] for l in gold]
            if pred_names == label_names: instance_matched = 1
            if sum(gold) == 0 and sum(pred) == 0:
                f1 = 1.0
            else:
                f1 = cal_f1(pred_names, label_names, {v: k for k, v in label_map.items()})
            if f1 >= max_f1:
                max_f1 = f1
                key = len(gold)

        if v['cluster_size'] > 1:
            if v['cluster'] not in em_cluster_agg:
                em_cluster_agg[v['cluster']] = 1
            if is_em == 0: em_cluster_agg[v['cluster']] = 0

            if v['cluster'] not in em_cluster_relaxed:
                em_cluster_relaxed[v['cluster']] = 1
            if instance_matched == 0: em_cluster_relaxed[v['cluster']] = 0

            if v['cluster'] not in f1_cluster_80:
                f1_cluster_80[v['cluster']] = 1
            if max_f1 < 0.8: f1_cluster_80[v['cluster']] = 0

        max_f1s.append(max_f1)
        em_counter += instance_matched
        f1_dist[key].append(max_f1)

    assert len(em_cluster_relaxed) == len(em_cluster_agg)
    assert len(f1_cluster_80) == len(em_cluster_agg)

    em_cluster_relaxed_res = sum(em_cluster_relaxed.values()) / len(em_cluster_relaxed)
    em_cluster_agg_res = sum(em_cluster_agg.values()) / len(em_cluster_agg)
    f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)

    label_names = [label_map[l] for l in all_golds]
    pred_names = [label_map[p] for p in all_preds]

    eval_pos_f1 = cal_f1(pred_names, label_names, {v: k for k, v in label_map.items()})

    print("the current eval positive class Micro F1 (Agg) is: %.4f" % eval_pos_f1)
    print("the current eval positive class Macro F1 (Relaxed) is: %.4f" % np.mean(max_f1s))
    print("the current eval positive class Macro F1 (Agg) is: %.4f" % np.mean(macro_f1s))

    print("the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(golds)))

    print("%d Clusters" % len(em_cluster_relaxed))
    print("the current eval clustered EM (Agg) is: %.4f" % (em_cluster_agg_res))
    print("the current eval clustered EM (Relaxed) is: %.4f" % (em_cluster_relaxed_res))
    print("the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res))

    return np.mean(max_f1s), (em_cluster_relaxed_res), (f1_cluster_80_res)

def main(args):
    labels_file = args.labels_file
    preds_file = args.preds_file
    metrics_output_file = args.metrics_output_file

    with open(labels_file) as infile:
        gold_answers = json.load(infile)

    with open(preds_file) as infile:
        pred_answers = json.load(infile)

    if len(gold_answers) != len(pred_answers):
        raise Exception("The prediction file does not contain the same number of lines as the "
                        "number of test instances.")

    f1, em, c = evaluate(gold_answers, pred_answers)
    results = {
        'F1': f1,
        'EM': em,
        'C': c
    }
    with open(metrics_output_file, "w") as f:
        f.write(json.dumps(results))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TORQUE predictions')
    # Required Parameters
    parser.add_argument('--labels_file', type=str, help='Location of test labels', default=None)
    parser.add_argument('--preds_file', type=str, help='Location of predictions', default=None)
    parser.add_argument('--metrics_output_file',
                        type=str,
                        help='Location of output metrics file',
                        default="metrics.json")

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)


