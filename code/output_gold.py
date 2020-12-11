from utils import *
from transformers import *

def output_gold(split):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
    data = load_data("../data/", "individual_%s" % split, "_end2end_final.json")
    features = convert_to_features_roberta(data, tokenizer, max_length=178,
                                           evaluation=True, instance=False, end_to_end=True)
    labels = select_field(features, 'label')

    # collect unique question ids for EM calculation
    question_ids = select_field(features, 'question_id')

    # collect unique question culster for EM-cluster calculation
    question_cluster = select_field(features, 'question_cluster')
    question_cluster_size = select_field(features, 'cluster_size')
    idv_answers = select_field(features, 'individual_answers')

    # questions
    questions = select_field(features, 'question')

    gold = {}
    for id, l, q, c, cs, idv in zip(question_ids, labels, questions, question_cluster,
                                    question_cluster_size, idv_answers):
        gold[id] = {'label': l,
                    'cluster': c,
                    'cluster_size': cs,
                    'idv_answers': idv}
        if len(gold) < 3:
            print(gold)

    with open("./output/%s_gold.json" % split, 'w') as outfile:
        json.dump(gold, outfile)
    return

output_gold('dev')