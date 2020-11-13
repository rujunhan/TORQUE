import json
import nltk

def spans(txt):
    tokens = nltk.word_tokenize(txt)
    offset = 0
    for token, tag in nltk.pos_tag(tokens):
        offset = txt.find(token, offset)
        yield token, offset, offset + len(token), tag
        offset += len(token)

def find_token_index(passage):
    tokens = spans(passage)
    token2idx = {}
    count = 0
    for token, start, end, tag in tokens:
        token2idx[(start, end)] = (token, count, tag)
        count += 1
    return token2idx

def fuzzy_match(tok, ans):
    # tok contains ans or ans contains tok
    ans = (int(ans.split(',')[0][1:]), int(ans.split(',')[1][:-1]))
    if tok[0] >= ans[0] and tok[1] <= ans[1]:
        return True
    elif ans[0] >= tok[0] and ans[1] <= tok[1]:
        return True
    else:
        return False

def remove_punct(ans):
    # remove punctuations in answer to match nltk tokenization
    # return it's updated span indices
    temp = ''.join([c for c in ans[1] if c.isalpha()])
    offset = ans[1].find(temp)
    key = x[0].split(',')
    key = (int(key[0][1:]) + offset, int(key[0][1:]) + offset + len(temp))
    return key

def sort_events(events):
    return [x[1] for x in (sorted([(int(e[0].split(',')[0][1:]), e) for e in events]))]

def compute_cluster_size(pid, cid):
    return len(question_cluster[pid][cid])

raw_path = "../raw/"

split = "test"

with open('%s/%s.json' % (raw_path, split), 'r') as infile:
    data = json.load(infile)

with open('%s/question_clustering.json' % raw_path, 'r') as infile:
    question_cluster = json.load(infile)

samples = {}
for k, passage in data.items():
    context = passage['passage'].replace("\"", "@")  # nltk has problem processing ". Replace it with "@"
    prefix = k

    count = 0
    for question in passage['question_answer_pairs']:

        cid = passage['question_answer_pairs'][question]['cluster_id']
        cluster = prefix + '_' + cid

        sample = {'context': nltk.word_tokenize(context),
                  'question': question,
                  'question_cluster': cluster,
                  'cluster_size': compute_cluster_size(prefix, cid)}

        samples["%s_%s" % (prefix, count)] = sample
        count += 1

print(len(samples))

with open('../data/individual_test_end2end_final.json', 'w') as outfile:
    json.dump(samples, outfile)