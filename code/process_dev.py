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

split = "dev"

with open('%s/%s.json' % (raw_path, split), 'r') as infile:
    data = json.load(infile)

with open('%s/question_clustering.json' % raw_path, 'r') as infile:
    question_cluster = json.load(infile)

samples = {}
for k, passage in data.items():
    context = passage['passage'].replace("\"", "@")  # nltk has problem processing ". Replace it with "@"
    token2idx = find_token_index(context)
    events = list(zip(passage['events']['answer']['indices'],
                      passage['events']['answer']['spans']))
    doc_id = '_'.join(passage['events']['passageID'].split('_')[1:-2])
    prefix = passage['events']['passageID']

    count = 0
    for question in passage['question_answer_pairs']:

        cid = passage['question_answer_pairs'][question]['cluster_id']
        cluster = prefix + '_' + cid

        sample = {'context': nltk.word_tokenize(context),
                  'question': question,
                  'question_cluster': cluster,
                  'cluster_size': compute_cluster_size(prefix, cid)}

        result = passage['question_answer_pairs'][question]

        answers = list(zip(result['answer']['indices'], result['answer']['spans']))
        ans = {}
        ignore_events, add_events = 0, 0
        ans_indices = []

        labels, types = [0] * len(token2idx), [0] * len(token2idx)
        assert len(token2idx) == len(labels)

        for x in sort_events(events):
            key = x[0].split(',')
            key = (int(key[0][1:]), int(key[1][:-1]))

            ## Step One
            if key not in token2idx:
                # case 1: 's
                if nltk.word_tokenize(x[1])[-1] == '\'s':
                    key = (key[1] - 2, key[1])
                    print("case 1: 's", key, token2idx[key])
                elif len(x[1]) == 1:
                    print("case 2: single character", x)
                    ignore_events += 1
                    continue
                else:
                    # case 3: ans in nltk after removing punctuations
                    new_key = remove_punct(x)
                    if new_key in token2idx:
                        print("case 3: correct after removing punct", key, new_key, token2idx[new_key])
                        key = new_key
                    # case 4: answer contained in nltk tokens or vice versa
                    elif sum([fuzzy_match(k, x[0]) for k, v in token2idx.items()]) > 0:
                        for k, v in token2idx.items():
                            if fuzzy_match(k, x[0]):
                                key = k
                                print("case 4: answer contained in nltk or vice versa", x, key, token2idx[k])
                                break
            ans_indices.append(token2idx[key][1])

            ## Step Two: fix answer if they are not nouns or verbs
            prev_list = ['be', 'is', 'am', 'are', 'were', 'was', 'been', 'turn', 'turns', 'turned',
                         'become', 'becomes', 'became', 'becoming', 'remain', 'remaining' 'remained', 'remains',
                         'go', 'goes', 'went', 'going', 'stay', 'stays', 'staying', 'stayed']

            if token2idx[key][2][0] not in ['N', 'V']:
                if token2idx[key][2] in ['JJ', 'IN', 'RB', 'RBR']:
                    fixable = False
                    for k, v in token2idx.items():
                        if v[1] < token2idx[key][1]:
                            if v[1] + 1 == token2idx[key][1]:
                                if ((v[0] in prev_list and token2idx[key][2] in ['JJ', 'RB', 'RBR']) or
                                        v[0] in prev_list[:7] and token2idx[key][2] in ['IN']):
                                    if v[1] not in ans_indices:
                                        print('fix case 1:', token2idx[key])
                                        fixable = True
                                        types[v[1]] = 1
                                        if x in answers: labels[v[1]] = 1
                                        add_events += 1
                            if v[1] + 2 == token2idx[key][1]:
                                if v[0] in prev_list and token2idx[key][2] == 'JJ':
                                    if v[1] not in ans_indices:
                                        print('fix case 2:', token2idx[key])
                                        fixable = True
                                        types[v[1]] = 1
                                        if x in answers: labels[v[1]] = 1
                                        add_events += 1
                                        break
                    if not fixable: ignore_events += 1
                else:
                    print("ignore case:", token2idx[key][2])
                    ignore_events += 1
            else:
                types[token2idx[key][1]] = 1
                if x in answers: labels[token2idx[key][1]] = 1

        assert sum(types) + ignore_events == len(events)
        assert sum(types) >= sum(labels)

        sample['answers'] = {'labels': labels, 'types': types}

        if split in ['dev', 'test']:
        ######################### Individual Answers ##############################
            sample['individual_answers'] = []
            for i, a in enumerate(result['individual_answers']):
                answers = list(zip(a['indices'], a['spans']))
                ans = {}
                ans_indices = []
                labels, types = [0] * len(token2idx), [0] * len(token2idx)
                assert len(token2idx) == len(labels)
                for x in sort_events(events):
                    key = x[0].split(',')
                    key = (int(key[0][1:]), int(key[1][:-1]))

                    # Step One
                    if key not in token2idx:
                        # case 1: 's
                        if nltk.word_tokenize(x[1])[-1] == '\'s':
                            key = (key[1] - 2, key[1])
                            print("case 1: 's", key, token2idx[key])
                        elif len(x[1]) == 1:
                            print("case 2: single character", x)
                            ignore_events += 1
                            continue
                        else:
                            # case 3: ans in nltk after removing punctuations
                            new_key = remove_punct(x)
                            if new_key in token2idx:
                                print("case 3: correct after removing punct", key, new_key, token2idx[new_key])
                                key = new_key
                            # case 4: answer contained in nltk tokens or vice versa
                            elif sum([fuzzy_match(k, x[0]) for k,v in token2idx.items()]) > 0:
                                for k, v in token2idx.items():
                                    if fuzzy_match(k, x[0]):
                                        key = k
                                        print("case 4: answer contained in nltk or vice versa", x, key, token2idx[k])
                                        break

                    ans_indices.append(token2idx[key][1])
                    ## Step Two: fix answer if they are not nouns or verbs
                    prev_list = ['be', 'is', 'am', 'are', 'were', 'was', 'been', 'turn', 'turns', 'turned',
                                'become', 'becomes', 'became', 'becoming', 'remain', 'remaining' 'remained', 'remains',
                                'go', 'goes', 'went', 'going', 'stay', 'stays', 'staying', 'stayed']

                    if token2idx[key][2][0] not in ['N', 'V']:
                        if token2idx[key][2] in ['JJ', 'IN', 'RB', 'RBR']:
                            fixable = False
                            for k, v in token2idx.items():
                                if v[1] < token2idx[key][1]:
                                    if v[1] + 1 == token2idx[key][1]:
                                        if ((v[0] in prev_list and token2idx[key][2] in ['JJ', 'RB', 'RBR']) or
                                            v[0] in prev_list[:7] and token2idx[key][2] in ['IN']):
                                            if v[1] not in ans_indices:
                                                print('fix case 1:', token2idx[key])
                                                fixable = True
                                                types[v[1]] = 1
                                                if x in answers: labels[v[1]] = 1
                                                add_events += 1
                                    if v[1] + 2 == token2idx[key][1]:
                                        if v[0] in prev_list and token2idx[key][2] == 'JJ':
                                            if v[1] not in ans_indices:
                                                print('fix case 2:', token2idx[key])
                                                fixable = True
                                                types[v[1]] = 1
                                                if x in answers: labels[v[1]] = 1
                                                add_events += 1
                                                break
                            if not fixable: ignore_events += 1
                        else:
                            print("ignore case:", token2idx[key][2])
                            ignore_events += 1
                    else:
                        types[token2idx[key][1]] = 1
                        if x in answers: labels[token2idx[key][1]] = 1
                sample['individual_answers'].append({'labels': labels, 'types': types})
                if sum(labels) == 0:
                    break

        samples["%s_%s" % (prefix, count)] = sample
        count += 1

print(len(samples))

with open('../data/individual_dev_end2end_final.json', 'w') as outfile:
    json.dump(samples, outfile)