
import os
import time
import random
import re
import sys
import string

from utils.utils import *

NEW_ENT_TYPE = ['Collection', 'Company', 'Software', 'Info-Type']

WLP_ENT_NAME = ['Amount', 'Reagent', 'Device', 'Time', 'Speed', 'Action', 'Mention', 'Location', 'Numerical',
                'Method', 'Temperature', 'Modifier', 'Concentration', 'Size', 'Generic-Measure', 'Seal',
                'Measure-Type', 'Misc', 'pH', 'Unit']

PubMed_ENT_NAME = ['Info-Type', 'Method', 'Reagent', 'Action', 'Software', 'Numerical', 'Modifier', 'Collection',
                   'Company', 'Amount', 'Mention', 'Generic-Measure', 'Location', 'Time', 'Temperature', 'Measure-Type',
                   'Device', 'Concentration', 'Size', 'pH', 'Speed', 'Seal', 'Misc', 'Unit']

ChemSyn_ENT_NAME = ['Info-Type', 'Method', 'Reagent', 'Action', 'Software', 'Numerical', 'Modifier', 'Collection',
                    'Company', 'Amount', 'Mention', 'Generic-Measure', 'Location', 'Time', 'Temperature', 'Measure-Type',
                    'Device', 'Concentration', 'Size', 'pH', 'Speed', 'Seal', 'Misc', 'Unit']

ENT_NAME = {
    'wlp': WLP_ENT_NAME,
    'pubmed': PubMed_ENT_NAME,
    'chemsyn': ChemSyn_ENT_NAME,
}

ENT_END = {
    'wlp': [("[ARG1-" + item + "-END]").lower() for item in WLP_ENT_NAME] +
           [("[ARG2-" + item + "-END]").lower() for item in WLP_ENT_NAME],
    'pubmed': [("[ARG1-" + item + "-END]").lower() for item in PubMed_ENT_NAME] +
              [("[ARG2-" + item + "-END]").lower() for item in PubMed_ENT_NAME],
    'chemsyn': [("[ARG1-" + item + "-END]").lower() for item in ChemSyn_ENT_NAME] +
               [("[ARG2-" + item + "-END]").lower() for item in ChemSyn_ENT_NAME],
}

ENT_START = {
    'wlp': [("[ARG1-" + item + "-START]").lower() for item in WLP_ENT_NAME] +
           [("[ARG2-" + item + "-START]").lower() for item in WLP_ENT_NAME],
    'pubmed': [("[ARG1-" + item + "-START]").lower() for item in PubMed_ENT_NAME] +
              [("[ARG2-" + item + "-START]").lower() for item in PubMed_ENT_NAME],
    'chemsyn': [("[ARG1-" + item + "-START]").lower() for item in ChemSyn_ENT_NAME] +
               [("[ARG2-" + item + "-START]").lower() for item in ChemSyn_ENT_NAME],
}

NO_RELATION = "no_relation"

WLP_REL_NAME = ['no_relation', 'Acts-on', 'Measure', 'Count', 'Creates', 'Using', 'Site',
                'Or', 'Product', 'Setting', 'Coreference-Link', 'Mod-Link', 'Meronym',
                'Measure-Type-Link', 'Commands', 'Misc-Link', 'Of-Type']

PubMed_REL_NAME = ['no_relation', 'Acts-on', 'Site', 'Using', 'Product', 'Coreference-Link',
                   'Meronym', 'Setting', 'Measure', 'Mod-Link', 'Belong-To', 'Measure-Type-Link',
                   'Or', 'Count', 'Of-Type', 'Commands', 'Creates', 'Misc-Link']

ChemSyn_REL_NAME = ['no_relation', 'Acts-on', 'Site', 'Using', 'Product', 'Coreference-Link',
                    'Meronym', 'Setting', 'Measure', 'Mod-Link', 'Belong-To', 'Measure-Type-Link',
                    'Or', 'Count', 'Of-Type', 'Commands', 'Creates', 'Misc-Link']

REL_NAME = {
    'wlp': WLP_REL_NAME,
    'pubmed': PubMed_REL_NAME,
    'chemsyn': ChemSyn_REL_NAME,
}

REL2IDX = {
    'wlp': dict([(label, id) for id, label in enumerate(REL_NAME['wlp'])]),
    'pubmed': dict([(label, id) for id, label in enumerate(REL_NAME['pubmed'])]),
    'chemsyn': dict([(label, id) for id, label in enumerate(REL_NAME['chemsyn'])]),
}

ENT_ID = [("[T" + str(i) + "]").lower() for i in range(2000)]

def load_from_txt(data_path, verbose=False, strip=True):
    examples = []

    with open(data_path, encoding='utf-8') as infile:
        while True:
            line = infile.readline()
            if len(line) == 0:
                break

            if strip:
                line = line.strip()

            examples.append(line)

    if verbose:
        print("{} examples read in {} .".format(len(examples), data_path))
    return examples


def decompose_tokenized_text(token_list):
    """
    Split the output of BERT tokenizer into word_list and tag_list
    """

    # The initial tag is 'O'
    tag = 'O'
    word_list = []
    tag_list = []

    # A flag to indicate if it is in the entity now
    # 0 means out, 1 means start, 2 means in the middle
    in_ent = 0

    for token in token_list:

        if token == '':
            continue

        # if the token ends with '-START]'
        # change the ent flag
        # if token[-len('-START]'):] == '-START]':
        if token.endswith('-START]'):
            tag = token[1:-1].rsplit('-', 1)[0]
            in_ent = 1
        # elif token[-len('-END]'):] == '-END]':
        elif token.endswith('-END]'):
            tag = 'O'
            in_ent = 0
        else:
            word_list.append(token)

            if in_ent == 1:
                tag_list.append('B-' + tag)
                in_ent = 2
            elif in_ent == 2:
                tag_list.append('I-' + tag)
            else:
                tag_list.append(tag)

    return word_list, tag_list


def index_ent_in_sentence(word_list, tag_list):
    ent_queue, ent_idx_queue, ent_type_queue = [], [], []
    ent_list, ent_idx_list, ent_type_list = [], [], []

    for word_idx in range(len(word_list)):

        if 'B-' in tag_list[word_idx]:
            if ent_queue:
                ent_list.append(' '.join(ent_queue).strip())
                ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1] + 1))

                assert len(set(ent_type_queue)) == 1
                ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []
            ent_queue.append(word_list[word_idx])
            ent_idx_queue.append(word_idx)
            ent_type_queue.append(tag_list[word_idx][2:])

        if 'I-' in tag_list[word_idx]:
            ent_queue.append(word_list[word_idx])
            ent_idx_queue.append(word_idx)
            ent_type_queue.append(tag_list[word_idx][2:])

        if 'O' == tag_list[word_idx] or word_idx == len(word_list) - 1:
            if ent_queue:
                ent_list.append(' '.join(ent_queue).strip())
                ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1] + 1))

                assert len(set(ent_type_queue)) == 1
                ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []

    return ent_list, ent_idx_list, ent_type_list


def mask_entities(tokens, entity_offsets, subj_entity_start, subj_entity_end,
                  obj_entity_start, obj_entity_end):
    subj_entity, obj_entity = entity_offsets

    if subj_entity[0] < obj_entity[0]:
        tokens = tokens[:subj_entity[0]] + [subj_entity_start] + tokens[subj_entity[0]:subj_entity[1]] + \
                 [subj_entity_end] + tokens[subj_entity[1]:obj_entity[0]] + [obj_entity_start] + \
                 tokens[obj_entity[0]:obj_entity[1]] + [obj_entity_end] + tokens[obj_entity[1]:]

        subj_entity = (subj_entity[0] + 1, subj_entity[1] + 1)
        obj_entity = (obj_entity[0] + 3, obj_entity[1] + 3)

    else:
        tokens = tokens[:obj_entity[0]] + [obj_entity_start] + tokens[obj_entity[0]:obj_entity[1]] + \
                 [obj_entity_end] + tokens[obj_entity[1]:subj_entity[0]] + [subj_entity_start] + \
                 tokens[subj_entity[0]:subj_entity[1]] + [subj_entity_end] + tokens[subj_entity[1]:]

        obj_entity = (obj_entity[0] + 1, obj_entity[1] + 1)
        subj_entity = (subj_entity[0] + 3, subj_entity[1] + 3)

    return tokens, (subj_entity, obj_entity)


def score(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if verbose:
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def index_ent_in_prediction(word_list, tag_list):
    ent_queue, ent_idx_queue, ent_type_queue = [], [], []
    ent_list, ent_idx_list, ent_type_list = [], [], []

    for word_idx in range(len(word_list)):

        if 'B-' in tag_list[word_idx]:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    #                     ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []
            ent_queue.append(word_list[word_idx])
            ent_idx_queue.append(word_idx)
            ent_type_queue.append(tag_list[word_idx][2:])

        if 'I-' in tag_list[word_idx]:
            if word_idx == 0 or (word_idx > 0 and tag_list[word_idx][2:] == tag_list[word_idx - 1][2:]):
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])
            else:
                if ent_queue:

                    if len(set(ent_type_queue)) != 1:
                        print(ent_queue)
                        print(ent_idx_queue)
                        print(ent_type_queue)
                        print(Counter(ent_type_queue).most_common())
                        print()
                    else:
                        ent_list.append(' '.join(ent_queue).strip())
                        #                         ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                        ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                        assert len(set(ent_type_queue)) == 1
                        ent_type_list.append(ent_type_queue[0])

                ent_queue, ent_idx_queue, ent_type_queue = [], [], []
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])

        if 'O' == tag_list[word_idx] or word_idx == len(word_list) - 1:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    #                     ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]+1))
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []

    return ent_list, ent_idx_list, ent_type_list


def get_processed_sentences(data_name, data_class, max_len, batch_size,
                            tokenizer, down_sample=False, down_sample_rate=0.5):

    data_dir = DATA_DIR[data_name][data_class]
    print(f"\nLoad data from: {data_dir}")

    file_order = FILE_ORDER[data_name][data_class]
    # total_sen_num = int(budget / DATA_PRICE[data_name]) if budget else None
    rel2idx = REL2IDX[data_name]

    all_label_list = []
    prepared_sen_over_files = []
    prepared_ent_id_over_files = []

    sen_over_files = []
    pos_pairs_over_files = []
    neg_pairs_over_files = []

    txt_file_list = load_from_txt(file_order)

    sen_count = 0

    for txt_file in txt_file_list:

        prepared_sen_list = []
        prepared_ent_id_list = []

        ann_file = txt_file[:-3] + "ann"

        if not (os.path.isfile(f"{data_dir}/{txt_file}") and os.path.isfile(f"{data_dir}/{ann_file}")):
            print(f"{data_dir}/{txt_file}")
            print(f"{data_dir}/{ann_file}")
            continue

        sen_list = load_from_txt(f"{data_dir}/{txt_file}", strip=False)
        sen_len_list = [len(item) for item in sen_list]

        ann_list = load_from_txt(f"{data_dir}/{ann_file}")
        all_sen_str = ''.join(sen_list)

        ent_start_list = []

        arg1_rel_arg2_tuple_list = []

        intermedia_entity_dict = dict([(item.split('\t')[0],
                                        item.split('\t')[1].split(' ')[0].split(":")[1]) for item in ann_list \
                                       if item[0] == 'E'])
        for item in ann_list:

            # 'T' means the entity
            if item[0] == 'T':
                try:
                    ent_id, label_offset, ent_str = item.split('\t')
                except:
                    # print('item split problem')
                    # print(ann_file)
                    # print(item)
                    continue

                try:
                    if ';' not in label_offset:
                        ent_label, ent_start, ent_end = label_offset.split(' ')
                        ent_start, ent_end = int(ent_start), int(ent_end)
                        all_label_list.append(ent_label)
                    else:
                        continue
                except:
                    # print('label_offset split problem')
                    # print(label_offset)
                    continue

                assert ent_str == all_sen_str[ent_start:ent_end] or \
                       ent_str == all_sen_str[ent_start:ent_end].strip()

                ent_start_list.append((ent_start, (ent_str, ent_start, ent_end, ent_label, ent_id)))

            # 'E' means the 'Acts-on' relation
            # Handle 'Acts-on' relation here
            if item[0] == 'E':
                # print(item)

                try:
                    rel_id, action_rel_ent = item.split('\t')
                except:
                    # print('item split problem')
                    # print(item)
                    continue

                _, arg1_id = action_rel_ent.split(' ')[0].split(':')

                if len(action_rel_ent.split(' ')) == 1:
                    continue

                for item in action_rel_ent.split(' ')[1:]:
                    rel_str, arg2_id = item.split(':')

                    if arg2_id[0] == 'E':
                        if arg2_id not in intermedia_entity_dict:
                            print("not in intermedia_entity_dict")
                            print(rel_id, arg1_id, rel_str, arg2_id)
                            print()
                        else:
                            # print(rel_id, arg1_id, rel_str, arg2_id, intermedia_entity_dict[arg2_id])
                            arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, intermedia_entity_dict[arg2_id]))
                            # print(arg1_id, rel_str, intermedia_entity_dict[arg2_id])
                    else:
                        arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, arg2_id))

            # Handle other relations
            if item[0] == 'R':
                # print(item)

                try:
                    rel_id, action_rel_ent = item.split('\t')
                except:
                    # print('item split problem')
                    # print(item)
                    continue

                rel_str, arg1_id_str, arg2_id_str = action_rel_ent.split(' ')
                arg1_id = arg1_id_str[len('Arg1:'):]
                arg2_id = arg2_id_str[len('Arg2:'):]

                # Make sure the reagents start with 'T'
                if arg1_id[0] == 'E':
                    arg1_id = intermedia_entity_dict[arg1_id]

                if arg2_id[0] == 'E':
                    arg2_id = intermedia_entity_dict[arg2_id]

                arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, arg2_id))


        # Just to split entities by sentence
        sen_start_list = [sum(sen_len_list[:index]) for index in range(len(sen_len_list) + 1)]

        sen_idx = 0
        ent_idx = 0
        sen_ent_dict = defaultdict(list)

        sorted_ent_start_list = sorted(ent_start_list, key=lambda x: x[0])

        while ent_idx < len(sorted_ent_start_list) and sen_idx < len(sen_start_list):
            #     print(ent_idx, sen_idx)

            ent_start, ent_info = sorted_ent_start_list[ent_idx]

            if ent_start >= sen_start_list[sen_idx] and \
                    ent_start < sen_start_list[sen_idx + 1]:

                ent_str, ent_start, ent_end, ent_label, ent_id = ent_info

                if ent_label in NEW_ENT_TYPE:
                    ent_idx += 1
                    continue

                # Remove the sentence offset
                ent_start, ent_end = ent_start - sen_start_list[sen_idx], \
                                     ent_end - sen_start_list[sen_idx]

                sen_ent_dict[sen_idx].append((ent_str, ent_start, ent_end, ent_label, ent_id))

                ent_idx += 1
                continue

            elif ent_start >= sen_start_list[sen_idx + 1]:
                sen_idx += 1
            else:
                print("Bug here")

        # # Control the total sentence number
        # if total_sen_num:
        #
        #     if sen_count >= total_sen_num:
        #         break
        #     else:
        #         if sen_count + len(sen_list) <= total_sen_num:
        #             select_num = len(sen_list)
        #         else:
        #             select_num = total_sen_num - sen_count
        # else:
        #     select_num = len(sen_list)
        #
        sen_count += len(sen_list)

        # Find the entity position in each sentence
        # for sen_idx in range(select_num):
        for sen_idx in range(len(sen_list)):

            sen_str = sen_list[sen_idx]

            # If the sentence doesn't contain any entity
            if sen_idx not in sen_ent_dict:
                prepared_sen_list.append(sen_str.strip())
                prepared_ent_id_list.append([])
                continue

            ent_list = sen_ent_dict[sen_idx]

            span_start, span_end = 0, 0
            span_list = []
            label_list = []
            ent_id_list = [(item[-1], item[0], item[-2]) for item in ent_list]

            for ent_str, ent_start, ent_end, ent_label, ent_id in ent_list:

                if ent_start > 0:
                    span_end = ent_start

                    if sen_str[span_start:span_end].strip():
                        span_list.append(sen_str[span_start:span_end])
                        label_list.append('O')

                span_list.append(sen_str[ent_start:ent_end])
                label_list.append(ent_label)

                span_start = ent_end

            # Add the last part of sentence
            if span_start != len(sen_str.strip()):
                span_end = len(sen_str.strip())
                span_list.append(sen_str[span_start:span_end])
                label_list.append('O')

            # Get the label for corresponding span
            span_label_list = list([item for item in zip(span_list, label_list) if item[0].strip()])

            # Add label to the sentence for bert tokenizer
            span_modified_list = []
            for span, span_label in span_label_list:
                if span_label != 'O':
                    span_modified_list += ['[{}-START]'.format(span_label), span, \
                                           '[{}-END]'.format(span_label)]
                else:
                    span_modified_list.append(span)

            prepared_sen = ' '.join(span_modified_list)

            prepared_sen_list.append(prepared_sen)
            prepared_ent_id_list.append(ent_id_list)

        prepared_sen_over_files.append(prepared_sen_list)
        prepared_ent_id_over_files.append(prepared_ent_id_list)

        ent_id_str_idx_type_list = []

        final_sen_list = []

        # Link the entitty to its id so that we can build up the (arg1, rel, arg2) tuple later
        for case_idx, (each_sen, each_ent_list) in enumerate(zip(prepared_sen_list, prepared_ent_id_list)):

            ent_id_list = [item[0] for item in each_ent_list]
            ent_str_org_list = [re.sub(' +', ' ', item[1]) for item in each_ent_list]
            ent_type_org_list = [item[2] for item in each_ent_list]

            tmp_word, tmp_tag = decompose_tokenized_text(each_sen.split(' '))
            ent_list, ent_idx_list, ent_type_list = index_ent_in_sentence(tmp_word, tmp_tag)

            final_sen_list.append(' '.join(tmp_word))

            if len(ent_list) != len(each_ent_list):
                print(case_idx)
                print("tmp_word, tmp_tag: ", tmp_word, tmp_tag, '\n')
                print("each_sen: ", each_sen, '\n')
                print(len(ent_list), len(each_ent_list), '\n')
                print("ent_list: ", ent_list, '\n')
                print("each_ent_list: ", each_ent_list, '\n')

            assert len(ent_list) == len(each_ent_list)

            if ent_list != ent_str_org_list:
                print("ent_list: ", ent_list)
                print("ent_str_org_list: ", ent_str_org_list)
                print(list([item1 == item2 for item1, item2 in zip(ent_list, ent_str_org_list)]))

            assert ent_list == ent_str_org_list

            if ent_type_org_list != ent_type_list:
                print(each_sen.split(' '))
                print(tmp_word, tmp_tag)
                print("ent_type_org_list: ", ent_type_org_list)
                print("ent_type_list: ", ent_type_list)

            assert ent_type_org_list == ent_type_list

            ent_id_str_idx_type_list.append(list(zip(ent_id_list, ent_list,
                                                     ent_idx_list, ent_type_list,
                                                     [case_idx] * len(ent_id_list))))

        # Build the id to entity dict for each file
        ent_id_str_idx_type_dict = dict([(item1[0], item1) for item in ent_id_str_idx_type_list for item1 in item])

        ent2senid = dict([(item1[0], item1[-1]) for idx, item in enumerate(ent_id_str_idx_type_list) for item1 in item])

        senid2ent = defaultdict(list)
        for _, item in enumerate(ent_id_str_idx_type_list):
            for item1 in item:
                senid2ent[item1[-1]].append(ent_id_str_idx_type_dict[item1[0]])

        # Make sure each id links to distinct entity
        if len(set([(item1[0], idx) for idx, item in enumerate(ent_id_str_idx_type_list)
                    for item1 in item])) != len(ent2senid):
            print(txt_file)
            print([(item1[0], item1[-1]) for idx, item in enumerate(ent_id_str_idx_type_list) for item1 in item])
            print(ent2senid)
        assert len(set([(item1[0], idx) for idx, item in enumerate(ent_id_str_idx_type_list)
                        for item1 in item])) == len(ent2senid)

        # Get the positive data
        sen_id_rel_tuple_pos_dict = defaultdict(list)

        for arg1_id, rel_str, arg2_id in arg1_rel_arg2_tuple_list:

            if arg1_id not in ent2senid:
                # if arg1_id in ent_id_str_idx_type_dict:
                #     print(txt_file)
                #     print("The entity {} is not in the dictionary.".format(arg1_id))
                continue

            if arg2_id not in ent2senid:
                # if arg2_id in ent_id_str_idx_type_dict:
                #     print(txt_file)
                #     print("The entity {} is not in the dictionary.".format(arg2_id))
                continue

            if ent2senid[arg1_id] != ent2senid[arg2_id]:
                # print(txt_file)
                # print("Two entities not in the same sentence.", rel_str, arg1_id, arg2_id,
                #       ent2senid[arg1_id],
                #       ent2senid[arg2_id], '\n')
                continue

            assert arg1_id in ent2senid and arg2_id in ent2senid
            assert arg1_id in ent2senid and arg2_id in ent2senid and ent2senid[arg1_id] == ent2senid[arg2_id]

            if ent_id_str_idx_type_dict[arg1_id][3] in NEW_ENT_TYPE or \
               ent_id_str_idx_type_dict[arg2_id][3] in NEW_ENT_TYPE:
                continue

            sen_id_rel_tuple_pos_dict[ent2senid[arg2_id]].append((rel_str,
                                                                  ent_id_str_idx_type_dict[arg1_id],
                                                                  ent_id_str_idx_type_dict[arg2_id]))

        # Generate negative pairs
        sen_id_rel_tuple_neg_dict = defaultdict(list)

        # for sen_idx in range(select_num):
        for sen_idx in range(len(sen_list)):

            pos_pair_id_list = [item[1][0] + ' ' + item[2][0] for item in sen_id_rel_tuple_pos_dict[sen_idx]]

            neg_pair_id_list_all = [item1[0] + ' ' + item2[0] for item1 in senid2ent[sen_idx]
                                    for item2 in senid2ent[sen_idx]
                                    if item1 != item2 and
                                    item1[0] + ' ' + item2[0] not in pos_pair_id_list]

            random.seed(1234)
            neg_pair_id_list = neg_pair_id_list_all

            if not neg_pair_id_list:
                sen_id_rel_tuple_neg_dict[sen_idx] = neg_pair_id_list

            for item in neg_pair_id_list:
                arg1_id, arg2_id = item.split(' ')

                assert arg1_id in ent2senid and arg2_id in ent2senid
                assert arg1_id in ent2senid and arg2_id in ent2senid and ent2senid[arg1_id] == ent2senid[arg2_id]

                if ent_id_str_idx_type_dict[arg1_id][3] in NEW_ENT_TYPE or \
                   ent_id_str_idx_type_dict[arg2_id][3] in NEW_ENT_TYPE:
                    continue

                sen_id_rel_tuple_neg_dict[ent2senid[arg2_id]].append(('no_relation',
                                                                      ent_id_str_idx_type_dict[arg1_id],
                                                                      ent_id_str_idx_type_dict[arg2_id]))

        assert len(final_sen_list) == len(sen_id_rel_tuple_pos_dict) and \
               len(final_sen_list) == len(sen_id_rel_tuple_neg_dict)

        sen_over_files.append(final_sen_list)
        pos_pairs_over_files.append(sen_id_rel_tuple_pos_dict)
        neg_pairs_over_files.append(sen_id_rel_tuple_neg_dict)

    print(f"The number of selected senteces: {sen_count}")
    processed_sen_over_files = []

    for sen_list, pos_pairs_list, neg_pairs_list in zip(sen_over_files, pos_pairs_over_files, neg_pairs_over_files):

        processed_sen_list = []

        for sen_idx, sen_str in enumerate(sen_list):

            word_list = sen_str.split(' ')

            # Process positive data
            assert sorted(set([item[1] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]])) == \
                   sorted(set([item[2] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]]))

            sorted_ent_list = sorted(set([item[1] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]]),
                                     key=lambda x: x[2][0])

            def mask_all_entities(tokens, sorted_entity_list):

                ent_dict = dict([(item[2][0], item) for item in sorted_entity_list])

                word_idx = 0

                tagged_token_list = []

                while word_idx < len(tokens):

                    if word_idx not in ent_dict:
                        tagged_token_list.append(tokens[word_idx])
                        word_idx += 1
                    else:
                        ent_id, ent_str, (ent_start, ent_end), ent_type, _ = ent_dict[word_idx]

                        if ent_str.strip() != ' '.join(tokens[ent_start:ent_end]).strip():
                            print(ent_str, ' '.join(tokens[ent_start:ent_end]))

                        assert ent_str.strip() == ' '.join(tokens[ent_start:ent_end]).strip()

                        tagged_token_list.append("[arg1-" + ent_type.lower() + "-start]")
                        tagged_token_list.append("[" + ent_id.lower() + "]")
                        tagged_token_list.append(ent_str)
                        tagged_token_list.append("[arg1-" + ent_type.lower() + "-end]")
                        word_idx = ent_end

                return tagged_token_list

            processed_word_list = mask_all_entities(word_list, sorted_ent_list)

            processed_sen_list.append(f"{tokenizer.cls_token} {' '.join(processed_word_list)} {tokenizer.eos_token}")

        processed_sen_over_files.append(processed_sen_list)

    start_time = time.time()

    # tokenized_texts = [tokenizer.tokenize(sent) for sent in processed_sen_list][:500]
    tokenized_texts = [[tokenizer.tokenize(sent) for sent in processed_sen_list] for processed_sen_list in
                       processed_sen_over_files]

    print("--- %s seconds ---" % (time.time() - start_time))

    # Remove unrelated entities from tokenzied text
    purified_texts = []
    relation_list = []
    type_list = []
    arg1_ent_start_list = []
    arg2_ent_start_list = []

    for tokenized_sen_list, pos_pairs_list, neg_pairs_list in zip(tokenized_texts, pos_pairs_over_files,
                                                                  neg_pairs_over_files):

        for sen_idx, word_list in enumerate(tokenized_sen_list):

            assert len(tokenized_sen_list) == len(pos_pairs_list) and len(tokenized_sen_list) == len(neg_pairs_list)

            # Process positive data
            for relation, arg1, arg2 in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]:

                # Note that this is for speeding up the fa_me_re experiments
                if relation == "no_relation" and down_sample and random.random() > down_sample_rate:
                    continue

                arg1_id, arg1_str, arg1_offset, arg1_type, arg1_senid = arg1
                arg2_id, arg2_str, arg2_offset, arg2_type, arg2_senid = arg2

                arg1_id = arg1_id.lower()
                arg2_id = arg2_id.lower()

                if "[{}]".format(arg1_id) not in word_list or "[{}]".format(arg2_id) not in word_list:
                    print("[{}]".format(arg1_id))
                    print("[{}]".format(arg2_id))
                    print(word_list)

                assert "[{}]".format(arg1_id) in word_list
                assert "[{}]".format(arg2_id) in word_list

                arg1_ent_start = "[arg1-" + arg1_type + "-start]"
                arg1_ent_end = "[arg1-" + arg1_type + "-end]"

                arg2_ent_start = "[arg2-" + arg2_type + "-start]"
                arg2_ent_end = "[arg2-" + arg2_type + "-end]"

                arg1_ent_start = arg1_ent_start.lower()
                arg2_ent_start = arg2_ent_start.lower()

                arg1_ent_start_list.append(arg1_ent_start)
                arg2_ent_start_list.append(arg2_ent_start)

                def purify_word_list(tokens, arg1id, arg2id):

                    word_idx = 0
                    arg_tag = 0
                    purified_token_list = []

                    while word_idx < len(tokens):

                        if tokens[word_idx][-len('-start]'):] == '-start]':
                            if tokens[word_idx + 1] == '[' + arg1id + ']':
                                purified_token_list.append(tokens[word_idx])
                                arg_tag = 1
                            elif tokens[word_idx + 1] == '[' + arg2id + ']':
                                purified_token_list.append(
                                    tokens[word_idx][:len('[arg')] + '2' + tokens[word_idx][len('[arg1'):])
                                arg_tag = 2
                            word_idx += 2
                        elif tokens[word_idx][-len('-end]'):] == '-end]':
                            if arg_tag:
                                purified_token_list.append(
                                    tokens[word_idx][:len('[arg')] + str(arg_tag) + tokens[word_idx][len('[arg1'):])
                                arg_tag = 0
                            word_idx += 1
                        else:
                            purified_token_list.append(tokens[word_idx])
                            word_idx += 1

                    return purified_token_list

                purified_word_list = purify_word_list(word_list, arg1_id, arg2_id)
                purified_texts.append(purified_word_list)

                relation_list.append(relation.rstrip(string.digits))

    # print(Counter(relation_list).items())

    assert len(purified_texts) == len(arg1_ent_start_list) and len(purified_texts) == len(arg2_ent_start_list)

    # Get the input_ids and labels
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in purified_texts],
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", truncating="post", padding="post")

    # attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    labels = [rel2idx[l] for l in relation_list]

    # print(purified_texts[0])
    # print(relation_list[0])
    # print(arg1_ent_start_list[0])
    # print(arg2_ent_start_list[0])

    arg1_idx_list = [purified_texts[sen_idx].index(arg1_ent_start_list[sen_idx])
                     if purified_texts[sen_idx].index(arg1_ent_start_list[sen_idx]) < max_len else max_len - 1
                     for sen_idx in range(len(purified_texts))
                     ]

    arg2_idx_list = [purified_texts[sen_idx].index(arg2_ent_start_list[sen_idx])
                     if purified_texts[sen_idx].index(arg2_ent_start_list[sen_idx]) < max_len else max_len - 1
                     for sen_idx in range(len(purified_texts))
                     ]

    # print(len(input_ids), len(attention_masks), len(labels), len(arg1_idx_list), len(arg2_idx_list))

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)
    arg1_idx_list = torch.tensor(arg1_idx_list)
    arg2_idx_list = torch.tensor(arg2_idx_list)

    final_data = TensorDataset(input_ids, attention_masks, labels, arg1_idx_list, arg2_idx_list)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader


def get_processed_sentences_budget(data_name, data_class, budget, max_len, batch_size,
                                   tokenizer, down_sample=False, down_sample_rate=0.5):

    data_dir = DATA_DIR[data_name][data_class]
    print(f"\nLoad data from: {data_dir}")

    file_order = FILE_ORDER[data_name][data_class]
    total_sen_num = int(budget / DATA_PRICE[data_name]) if budget else None
    rel2idx = REL2IDX['wlp']

    all_label_list = []
    prepared_sen_over_files = []
    prepared_ent_id_over_files = []

    sen_over_files = []
    pos_pairs_over_files = []
    neg_pairs_over_files = []

    txt_file_list = load_from_txt(file_order)

    sen_count = 0

    for txt_file in txt_file_list:

        prepared_sen_list = []
        prepared_ent_id_list = []

        ann_file = txt_file[:-3] + "ann"

        if not (os.path.isfile(f"{data_dir}/{txt_file}") and os.path.isfile(f"{data_dir}/{ann_file}")):
            print(f"{data_dir}/{txt_file}")
            print(f"{data_dir}/{ann_file}")
            continue

        sen_list = load_from_txt(f"{data_dir}/{txt_file}", strip=False)
        sen_len_list = [len(item) for item in sen_list]

        ann_list = load_from_txt(f"{data_dir}/{ann_file}")
        all_sen_str = ''.join(sen_list)

        ent_start_list = []

        arg1_rel_arg2_tuple_list = []

        intermedia_entity_dict = dict([(item.split('\t')[0],
                                        item.split('\t')[1].split(' ')[0].split(":")[1]) for item in ann_list \
                                       if item[0] == 'E'])
        for item in ann_list:

            # 'T' means the entity
            if item[0] == 'T':
                try:
                    ent_id, label_offset, ent_str = item.split('\t')
                except:
                    # print('item split problem')
                    # print(ann_file)
                    # print(item)
                    continue

                try:
                    if ';' not in label_offset:
                        ent_label, ent_start, ent_end = label_offset.split(' ')
                        ent_start, ent_end = int(ent_start), int(ent_end)
                        all_label_list.append(ent_label)
                    else:
                        continue
                except:
                    # print('label_offset split problem')
                    # print(label_offset)
                    continue

                assert ent_str == all_sen_str[ent_start:ent_end] or \
                       ent_str == all_sen_str[ent_start:ent_end].strip()

                ent_start_list.append((ent_start, (ent_str, ent_start, ent_end, ent_label, ent_id)))

            # 'E' means the 'Acts-on' relation
            # Handle 'Acts-on' relation here
            if item[0] == 'E':
                # print(item)

                try:
                    rel_id, action_rel_ent = item.split('\t')
                except:
                    # print('item split problem')
                    # print(item)
                    continue

                _, arg1_id = action_rel_ent.split(' ')[0].split(':')

                if len(action_rel_ent.split(' ')) == 1:
                    continue

                for item in action_rel_ent.split(' ')[1:]:
                    rel_str, arg2_id = item.split(':')

                    if arg2_id[0] == 'E':
                        if arg2_id not in intermedia_entity_dict:
                            print("not in intermedia_entity_dict")
                            print(rel_id, arg1_id, rel_str, arg2_id)
                            print()
                        else:
                            # print(rel_id, arg1_id, rel_str, arg2_id, intermedia_entity_dict[arg2_id])
                            arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, intermedia_entity_dict[arg2_id]))
                            # print(arg1_id, rel_str, intermedia_entity_dict[arg2_id])
                    else:
                        arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, arg2_id))

            # Handle other relations
            if item[0] == 'R':
                # print(item)

                try:
                    rel_id, action_rel_ent = item.split('\t')
                except:
                    # print('item split problem')
                    # print(item)
                    continue

                rel_str, arg1_id_str, arg2_id_str = action_rel_ent.split(' ')
                arg1_id = arg1_id_str[len('Arg1:'):]
                arg2_id = arg2_id_str[len('Arg2:'):]

                # Make sure the reagents start with 'T'
                if arg1_id[0] == 'E':
                    arg1_id = intermedia_entity_dict[arg1_id]

                if arg2_id[0] == 'E':
                    arg2_id = intermedia_entity_dict[arg2_id]

                arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, arg2_id))


        # Just to split entities by sentence
        sen_start_list = [sum(sen_len_list[:index]) for index in range(len(sen_len_list) + 1)]

        sen_idx = 0
        ent_idx = 0
        sen_ent_dict = defaultdict(list)

        sorted_ent_start_list = sorted(ent_start_list, key=lambda x: x[0])

        while ent_idx < len(sorted_ent_start_list) and sen_idx < len(sen_start_list):
            #     print(ent_idx, sen_idx)

            ent_start, ent_info = sorted_ent_start_list[ent_idx]

            if ent_start >= sen_start_list[sen_idx] and \
                    ent_start < sen_start_list[sen_idx + 1]:

                ent_str, ent_start, ent_end, ent_label, ent_id = ent_info

                if ent_label in NEW_ENT_TYPE:
                    ent_idx += 1
                    continue

                # Remove the sentence offset
                ent_start, ent_end = ent_start - sen_start_list[sen_idx], \
                                     ent_end - sen_start_list[sen_idx]

                sen_ent_dict[sen_idx].append((ent_str, ent_start, ent_end, ent_label, ent_id))

                ent_idx += 1
                continue

            elif ent_start >= sen_start_list[sen_idx + 1]:
                sen_idx += 1
            else:
                print("Bug here")

        # Control the total sentence number
        if total_sen_num:

            if sen_count >= total_sen_num:
                break
            else:
                if sen_count + len(sen_list) <= total_sen_num:
                    select_num = len(sen_list)
                else:
                    select_num = total_sen_num - sen_count
        else:
            select_num = len(sen_list)

        sen_count += select_num

        # Find the entity position in each sentence
        for sen_idx in range(select_num):

            sen_str = sen_list[sen_idx]

            # If the sentence doesn't contain any entity
            if sen_idx not in sen_ent_dict:
                prepared_sen_list.append(sen_str.strip())
                prepared_ent_id_list.append([])
                continue

            ent_list = sen_ent_dict[sen_idx]

            span_start, span_end = 0, 0
            span_list = []
            label_list = []
            ent_id_list = [(item[-1], item[0], item[-2]) for item in ent_list]

            for ent_str, ent_start, ent_end, ent_label, ent_id in ent_list:

                if ent_start > 0:
                    span_end = ent_start

                    if sen_str[span_start:span_end].strip():
                        span_list.append(sen_str[span_start:span_end])
                        label_list.append('O')

                span_list.append(sen_str[ent_start:ent_end])
                label_list.append(ent_label)

                span_start = ent_end

            # Add the last part of sentence
            if span_start != len(sen_str.strip()):
                span_end = len(sen_str.strip())
                span_list.append(sen_str[span_start:span_end])
                label_list.append('O')

            # Get the label for corresponding span
            span_label_list = list([item for item in zip(span_list, label_list) if item[0].strip()])

            # Add label to the sentence for bert tokenizer
            span_modified_list = []
            for span, span_label in span_label_list:
                if span_label != 'O':
                    span_modified_list += ['[{}-START]'.format(span_label), span, \
                                           '[{}-END]'.format(span_label)]
                else:
                    span_modified_list.append(span)

            prepared_sen = ' '.join(span_modified_list)

            prepared_sen_list.append(prepared_sen)
            prepared_ent_id_list.append(ent_id_list)

        prepared_sen_over_files.append(prepared_sen_list)
        prepared_ent_id_over_files.append(prepared_ent_id_list)

        ent_id_str_idx_type_list = []

        final_sen_list = []

        # Link the entitty to its id so that we can build up the (arg1, rel, arg2) tuple later
        for case_idx, (each_sen, each_ent_list) in enumerate(zip(prepared_sen_list, prepared_ent_id_list)):

            ent_id_list = [item[0] for item in each_ent_list]
            ent_str_org_list = [re.sub(' +', ' ', item[1]) for item in each_ent_list]
            ent_type_org_list = [item[2] for item in each_ent_list]

            tmp_word, tmp_tag = decompose_tokenized_text(each_sen.split(' '))
            ent_list, ent_idx_list, ent_type_list = index_ent_in_sentence(tmp_word, tmp_tag)

            final_sen_list.append(' '.join(tmp_word))

            if len(ent_list) != len(each_ent_list):
                print(case_idx)
                print("tmp_word, tmp_tag: ", tmp_word, tmp_tag, '\n')
                print("each_sen: ", each_sen, '\n')
                print(len(ent_list), len(each_ent_list), '\n')
                print("ent_list: ", ent_list, '\n')
                print("each_ent_list: ", each_ent_list, '\n')

            assert len(ent_list) == len(each_ent_list)

            if ent_list != ent_str_org_list:
                print("ent_list: ", ent_list)
                print("ent_str_org_list: ", ent_str_org_list)
                print(list([item1 == item2 for item1, item2 in zip(ent_list, ent_str_org_list)]))

            assert ent_list == ent_str_org_list

            if ent_type_org_list != ent_type_list:
                print(each_sen.split(' '))
                print(tmp_word, tmp_tag)
                print("ent_type_org_list: ", ent_type_org_list)
                print("ent_type_list: ", ent_type_list)

            assert ent_type_org_list == ent_type_list

            ent_id_str_idx_type_list.append(list(zip(ent_id_list, ent_list,
                                                     ent_idx_list, ent_type_list,
                                                     [case_idx] * len(ent_id_list))))

        # Build the id to entity dict for each file
        ent_id_str_idx_type_dict = dict([(item1[0], item1) for item in ent_id_str_idx_type_list for item1 in item])

        ent2senid = dict([(item1[0], item1[-1]) for idx, item in enumerate(ent_id_str_idx_type_list) for item1 in item])

        senid2ent = defaultdict(list)
        for _, item in enumerate(ent_id_str_idx_type_list):
            for item1 in item:
                senid2ent[item1[-1]].append(ent_id_str_idx_type_dict[item1[0]])

        # Make sure each id links to distinct entity
        if len(set([(item1[0], idx) for idx, item in enumerate(ent_id_str_idx_type_list)
                    for item1 in item])) != len(ent2senid):
            print(txt_file)
            print([(item1[0], item1[-1]) for idx, item in enumerate(ent_id_str_idx_type_list) for item1 in item])
            print(ent2senid)
        assert len(set([(item1[0], idx) for idx, item in enumerate(ent_id_str_idx_type_list)
                        for item1 in item])) == len(ent2senid)

        # Get the positive data
        sen_id_rel_tuple_pos_dict = defaultdict(list)

        for arg1_id, rel_str, arg2_id in arg1_rel_arg2_tuple_list:

            if arg1_id not in ent2senid:
                # if arg1_id in ent_id_str_idx_type_dict:
                #     print(txt_file)
                #     print("The entity {} is not in the dictionary.".format(arg1_id))
                continue

            if arg2_id not in ent2senid:
                # if arg2_id in ent_id_str_idx_type_dict:
                #     print(txt_file)
                #     print("The entity {} is not in the dictionary.".format(arg2_id))
                continue

            if ent2senid[arg1_id] != ent2senid[arg2_id]:
                # print(txt_file)
                # print("Two entities not in the same sentence.", rel_str, arg1_id, arg2_id,
                #       ent2senid[arg1_id],
                #       ent2senid[arg2_id], '\n')
                continue

            assert arg1_id in ent2senid and arg2_id in ent2senid
            assert arg1_id in ent2senid and arg2_id in ent2senid and ent2senid[arg1_id] == ent2senid[arg2_id]

            if ent_id_str_idx_type_dict[arg1_id][3] in NEW_ENT_TYPE or \
               ent_id_str_idx_type_dict[arg2_id][3] in NEW_ENT_TYPE:
                continue

            sen_id_rel_tuple_pos_dict[ent2senid[arg2_id]].append((rel_str,
                                                                  ent_id_str_idx_type_dict[arg1_id],
                                                                  ent_id_str_idx_type_dict[arg2_id]))

        # Generate negative pairs
        sen_id_rel_tuple_neg_dict = defaultdict(list)

        for sen_idx in range(select_num):
        # for sen_idx in range(len(sen_list)):

            pos_pair_id_list = [item[1][0] + ' ' + item[2][0] for item in sen_id_rel_tuple_pos_dict[sen_idx]]

            neg_pair_id_list_all = [item1[0] + ' ' + item2[0] for item1 in senid2ent[sen_idx]
                                    for item2 in senid2ent[sen_idx]
                                    if item1 != item2 and
                                    item1[0] + ' ' + item2[0] not in pos_pair_id_list]

            random.seed(1234)
            neg_pair_id_list = neg_pair_id_list_all

            if not neg_pair_id_list:
                sen_id_rel_tuple_neg_dict[sen_idx] = neg_pair_id_list

            for item in neg_pair_id_list:
                arg1_id, arg2_id = item.split(' ')

                assert arg1_id in ent2senid and arg2_id in ent2senid
                assert arg1_id in ent2senid and arg2_id in ent2senid and ent2senid[arg1_id] == ent2senid[arg2_id]

                if ent_id_str_idx_type_dict[arg1_id][3] in NEW_ENT_TYPE or \
                   ent_id_str_idx_type_dict[arg2_id][3] in NEW_ENT_TYPE:
                    continue

                sen_id_rel_tuple_neg_dict[ent2senid[arg2_id]].append(('no_relation',
                                                                      ent_id_str_idx_type_dict[arg1_id],
                                                                      ent_id_str_idx_type_dict[arg2_id]))

        assert len(final_sen_list) == len(sen_id_rel_tuple_pos_dict) and \
               len(final_sen_list) == len(sen_id_rel_tuple_neg_dict)

        sen_over_files.append(final_sen_list)
        pos_pairs_over_files.append(sen_id_rel_tuple_pos_dict)
        neg_pairs_over_files.append(sen_id_rel_tuple_neg_dict)

    print(f"The number of selected senteces: {sen_count}")
    processed_sen_over_files = []

    for sen_list, pos_pairs_list, neg_pairs_list in zip(sen_over_files, pos_pairs_over_files, neg_pairs_over_files):

        processed_sen_list = []

        for sen_idx, sen_str in enumerate(sen_list):

            word_list = sen_str.split(' ')

            # Process positive data
            assert sorted(set([item[1] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]])) == \
                   sorted(set([item[2] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]]))

            sorted_ent_list = sorted(set([item[1] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]]),
                                     key=lambda x: x[2][0])

            def mask_all_entities(tokens, sorted_entity_list):

                ent_dict = dict([(item[2][0], item) for item in sorted_entity_list])

                word_idx = 0

                tagged_token_list = []

                while word_idx < len(tokens):

                    if word_idx not in ent_dict:
                        tagged_token_list.append(tokens[word_idx])
                        word_idx += 1
                    else:
                        ent_id, ent_str, (ent_start, ent_end), ent_type, _ = ent_dict[word_idx]

                        if ent_str.strip() != ' '.join(tokens[ent_start:ent_end]).strip():
                            print(ent_str, ' '.join(tokens[ent_start:ent_end]))

                        assert ent_str.strip() == ' '.join(tokens[ent_start:ent_end]).strip()

                        tagged_token_list.append("[arg1-" + ent_type.lower() + "-start]")
                        tagged_token_list.append("[" + ent_id.lower() + "]")
                        tagged_token_list.append(ent_str)
                        tagged_token_list.append("[arg1-" + ent_type.lower() + "-end]")
                        word_idx = ent_end

                return tagged_token_list

            processed_word_list = mask_all_entities(word_list, sorted_ent_list)

            processed_sen_list.append(f"{tokenizer.cls_token} {' '.join(processed_word_list)} {tokenizer.eos_token}")

        processed_sen_over_files.append(processed_sen_list)

    start_time = time.time()

    # tokenized_texts = [tokenizer.tokenize(sent) for sent in processed_sen_list][:500]
    tokenized_texts = [[tokenizer.tokenize(sent) for sent in processed_sen_list] for processed_sen_list in
                       processed_sen_over_files]

    print("--- %s seconds ---" % (time.time() - start_time))

    # Remove unrelated entities from tokenzied text
    purified_texts = []
    relation_list = []
    type_list = []
    arg1_ent_start_list = []
    arg2_ent_start_list = []

    for tokenized_sen_list, pos_pairs_list, neg_pairs_list in zip(tokenized_texts, pos_pairs_over_files,
                                                                  neg_pairs_over_files):

        for sen_idx, word_list in enumerate(tokenized_sen_list):

            assert len(tokenized_sen_list) == len(pos_pairs_list) and len(tokenized_sen_list) == len(neg_pairs_list)

            # Process positive data
            for relation, arg1, arg2 in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]:

                # Note that this is for speeding up the fa_me_re experiments
                if relation == "no_relation" and down_sample and random.random() > down_sample_rate:
                    continue

                arg1_id, arg1_str, arg1_offset, arg1_type, arg1_senid = arg1
                arg2_id, arg2_str, arg2_offset, arg2_type, arg2_senid = arg2

                arg1_id = arg1_id.lower()
                arg2_id = arg2_id.lower()

                if "[{}]".format(arg1_id) not in word_list or "[{}]".format(arg2_id) not in word_list:
                    print("[{}]".format(arg1_id))
                    print("[{}]".format(arg2_id))
                    print(word_list)

                assert "[{}]".format(arg1_id) in word_list
                assert "[{}]".format(arg2_id) in word_list

                arg1_ent_start = "[arg1-" + arg1_type + "-start]"
                arg1_ent_end = "[arg1-" + arg1_type + "-end]"

                arg2_ent_start = "[arg2-" + arg2_type + "-start]"
                arg2_ent_end = "[arg2-" + arg2_type + "-end]"

                arg1_ent_start = arg1_ent_start.lower()
                arg2_ent_start = arg2_ent_start.lower()

                arg1_ent_start_list.append(arg1_ent_start)
                arg2_ent_start_list.append(arg2_ent_start)

                def purify_word_list(tokens, arg1id, arg2id):

                    word_idx = 0
                    arg_tag = 0
                    purified_token_list = []

                    while word_idx < len(tokens):

                        if tokens[word_idx][-len('-start]'):] == '-start]':
                            if tokens[word_idx + 1] == '[' + arg1id + ']':
                                purified_token_list.append(tokens[word_idx])
                                arg_tag = 1
                            elif tokens[word_idx + 1] == '[' + arg2id + ']':
                                purified_token_list.append(
                                    tokens[word_idx][:len('[arg')] + '2' + tokens[word_idx][len('[arg1'):])
                                arg_tag = 2
                            word_idx += 2
                        elif tokens[word_idx][-len('-end]'):] == '-end]':
                            if arg_tag:
                                purified_token_list.append(
                                    tokens[word_idx][:len('[arg')] + str(arg_tag) + tokens[word_idx][len('[arg1'):])
                                arg_tag = 0
                            word_idx += 1
                        else:
                            purified_token_list.append(tokens[word_idx])
                            word_idx += 1

                    return purified_token_list

                purified_word_list = purify_word_list(word_list, arg1_id, arg2_id)
                purified_texts.append(purified_word_list)

                relation_list.append(relation.rstrip(string.digits))

    # print(Counter(relation_list).items())

    assert len(purified_texts) == len(arg1_ent_start_list) and len(purified_texts) == len(arg2_ent_start_list)

    # Get the input_ids and labels
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in purified_texts],
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", truncating="post", padding="post")

    # attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    labels = [rel2idx[l] for l in relation_list]

    # print(purified_texts[0])
    # print(relation_list[0])
    # print(arg1_ent_start_list[0])
    # print(arg2_ent_start_list[0])

    arg1_idx_list = [purified_texts[sen_idx].index(arg1_ent_start_list[sen_idx])
                     if purified_texts[sen_idx].index(arg1_ent_start_list[sen_idx]) < max_len else max_len - 1
                     for sen_idx in range(len(purified_texts))
                     ]

    arg2_idx_list = [purified_texts[sen_idx].index(arg2_ent_start_list[sen_idx])
                     if purified_texts[sen_idx].index(arg2_ent_start_list[sen_idx]) < max_len else max_len - 1
                     for sen_idx in range(len(purified_texts))
                     ]

    # print(len(input_ids), len(attention_masks), len(labels), len(arg1_idx_list), len(arg2_idx_list))

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)
    arg1_idx_list = torch.tensor(arg1_idx_list)
    arg2_idx_list = torch.tensor(arg2_idx_list)

    final_data = TensorDataset(input_ids, attention_masks, labels, arg1_idx_list, arg2_idx_list)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader


def get_processed_sentences_da_budget(data_name_list, data_class, budget, max_len, batch_size, tokenizer, alpha=1,
                                      down_sample=False, down_sample_rate=0.5):

    src_data, tgt_data = data_name_list

    total_sen_num = int(budget / DATA_PRICE[tgt_data]) if budget else None
    rel2idx = REL2IDX['wlp']

    if not tgt_data:
        raise ValueError("There must be a target domain")

    if total_sen_num:
        print(f"Select top {total_sen_num} sentences from the target domain.")

    def load_from_dir(data_dir, file_order, total_sen_num=None):

        print(f"\nLoad data from: {data_dir}")
        all_label_list = []
        prepared_sen_over_files = []
        prepared_ent_id_over_files = []

        sen_over_files = []
        pos_pairs_over_files = []
        neg_pairs_over_files = []

        txt_file_list = load_from_txt(file_order)

        sen_count = 0

        for txt_file in txt_file_list:

            prepared_sen_list = []
            prepared_ent_id_list = []

            ann_file = txt_file[:-3] + "ann"

            if not (os.path.isfile(f"{data_dir}/{txt_file}") and os.path.isfile(f"{data_dir}/{ann_file}")):
                print(f"{data_dir}/{txt_file}")
                print(f"{data_dir}/{ann_file}")
                continue

            sen_list = load_from_txt(f"{data_dir}/{txt_file}", strip=False)
            sen_len_list = [len(item) for item in sen_list]

            ann_list = load_from_txt(f"{data_dir}/{ann_file}")
            all_sen_str = ''.join(sen_list)

            ent_start_list = []

            arg1_rel_arg2_tuple_list = []

            intermedia_entity_dict = dict([(item.split('\t')[0],
                                            item.split('\t')[1].split(' ')[0].split(":")[1]) for item in ann_list \
                                           if item[0] == 'E'])
            for item in ann_list:

                # 'T' means the entity
                if item[0] == 'T':
                    try:
                        ent_id, label_offset, ent_str = item.split('\t')
                    except:
                        # print('item split problem')
                        # print(ann_file)
                        # print(item)
                        continue

                    try:
                        if ';' not in label_offset:
                            ent_label, ent_start, ent_end = label_offset.split(' ')
                            ent_start, ent_end = int(ent_start), int(ent_end)
                            all_label_list.append(ent_label)
                        else:
                            continue
                    except:
                        # print('label_offset split problem')
                        # print(label_offset)
                        continue

                    assert ent_str == all_sen_str[ent_start:ent_end] or \
                           ent_str == all_sen_str[ent_start:ent_end].strip()

                    ent_start_list.append((ent_start, (ent_str, ent_start, ent_end, ent_label, ent_id)))

                # 'E' means the 'Acts-on' relation
                # Handle 'Acts-on' relation here
                if item[0] == 'E':
                    # print(item)

                    try:
                        rel_id, action_rel_ent = item.split('\t')
                    except:
                        # print('item split problem')
                        # print(item)
                        continue

                    _, arg1_id = action_rel_ent.split(' ')[0].split(':')

                    if len(action_rel_ent.split(' ')) == 1:
                        continue

                    for item in action_rel_ent.split(' ')[1:]:
                        rel_str, arg2_id = item.split(':')

                        if arg2_id[0] == 'E':
                            if arg2_id not in intermedia_entity_dict:
                                print("not in intermedia_entity_dict")
                                print(rel_id, arg1_id, rel_str, arg2_id)
                                print()
                            else:
                                # print(rel_id, arg1_id, rel_str, arg2_id, intermedia_entity_dict[arg2_id])
                                arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, intermedia_entity_dict[arg2_id]))
                                # print(arg1_id, rel_str, intermedia_entity_dict[arg2_id])
                        else:
                            arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, arg2_id))

                # Handle other relations
                if item[0] == 'R':
                    # print(item)

                    try:
                        rel_id, action_rel_ent = item.split('\t')
                    except:
                        # print('item split problem')
                        # print(item)
                        continue

                    rel_str, arg1_id_str, arg2_id_str = action_rel_ent.split(' ')
                    arg1_id = arg1_id_str[len('Arg1:'):]
                    arg2_id = arg2_id_str[len('Arg2:'):]

                    # Make sure the reagents start with 'T'
                    if arg1_id[0] == 'E':
                        arg1_id = intermedia_entity_dict[arg1_id]

                    if arg2_id[0] == 'E':
                        arg2_id = intermedia_entity_dict[arg2_id]

                    arg1_rel_arg2_tuple_list.append((arg1_id, rel_str, arg2_id))

            # Just to split entities by sentence
            sen_start_list = [sum(sen_len_list[:index]) for index in range(len(sen_len_list) + 1)]

            sen_idx = 0
            ent_idx = 0
            sen_ent_dict = defaultdict(list)

            sorted_ent_start_list = sorted(ent_start_list, key=lambda x: x[0])

            while ent_idx < len(sorted_ent_start_list) and sen_idx < len(sen_start_list):
                #     print(ent_idx, sen_idx)

                ent_start, ent_info = sorted_ent_start_list[ent_idx]

                if ent_start >= sen_start_list[sen_idx] and \
                        ent_start < sen_start_list[sen_idx + 1]:

                    ent_str, ent_start, ent_end, ent_label, ent_id = ent_info

                    if ent_label in NEW_ENT_TYPE:
                        ent_idx += 1
                        continue

                    # Remove the sentence offset
                    ent_start, ent_end = ent_start - sen_start_list[sen_idx], \
                                         ent_end - sen_start_list[sen_idx]

                    sen_ent_dict[sen_idx].append((ent_str, ent_start, ent_end, ent_label, ent_id))

                    ent_idx += 1
                    continue

                elif ent_start >= sen_start_list[sen_idx + 1]:
                    sen_idx += 1
                else:
                    print("Bug here")

            # Control the total sentence number
            if total_sen_num:

                if sen_count >= total_sen_num:
                    break
                else:
                    if sen_count + len(sen_list) <= total_sen_num:
                        select_num = len(sen_list)
                    else:
                        select_num = total_sen_num - sen_count
            else:
                select_num = len(sen_list)

            sen_count += select_num

            # Find the entity position in each sentence
            for sen_idx in range(select_num):

                sen_str = sen_list[sen_idx]

                # If the sentence doesn't contain any entity
                if sen_idx not in sen_ent_dict:
                    prepared_sen_list.append(sen_str.strip())
                    prepared_ent_id_list.append([])
                    continue

                ent_list = sen_ent_dict[sen_idx]

                span_start, span_end = 0, 0
                span_list = []
                label_list = []
                ent_id_list = [(item[-1], item[0], item[-2]) for item in ent_list]

                for ent_str, ent_start, ent_end, ent_label, ent_id in ent_list:

                    if ent_start > 0:
                        span_end = ent_start

                        if sen_str[span_start:span_end].strip():
                            span_list.append(sen_str[span_start:span_end])
                            label_list.append('O')

                    span_list.append(sen_str[ent_start:ent_end])
                    label_list.append(ent_label)

                    span_start = ent_end

                # Add the last part of sentence
                if span_start != len(sen_str.strip()):
                    span_end = len(sen_str.strip())
                    span_list.append(sen_str[span_start:span_end])
                    label_list.append('O')

                # Get the label for corresponding span
                span_label_list = list([item for item in zip(span_list, label_list) if item[0].strip()])

                # Add label to the sentence for bert tokenizer
                span_modified_list = []
                for span, span_label in span_label_list:
                    if span_label != 'O':
                        span_modified_list += ['[{}-START]'.format(span_label), span, \
                                               '[{}-END]'.format(span_label)]
                    else:
                        span_modified_list.append(span)

                prepared_sen = ' '.join(span_modified_list)

                prepared_sen_list.append(prepared_sen)
                prepared_ent_id_list.append(ent_id_list)

            prepared_sen_over_files.append(prepared_sen_list)
            prepared_ent_id_over_files.append(prepared_ent_id_list)

            ent_id_str_idx_type_list = []

            final_sen_list = []

            # Link the entitty to its id so that we can build up the (arg1, rel, arg2) tuple later
            for case_idx, (each_sen, each_ent_list) in enumerate(zip(prepared_sen_list, prepared_ent_id_list)):

                ent_id_list = [item[0] for item in each_ent_list]
                ent_str_org_list = [re.sub(' +', ' ', item[1]) for item in each_ent_list]
                ent_type_org_list = [item[2] for item in each_ent_list]

                tmp_word, tmp_tag = decompose_tokenized_text(each_sen.split(' '))
                ent_list, ent_idx_list, ent_type_list = index_ent_in_sentence(tmp_word, tmp_tag)

                final_sen_list.append(' '.join(tmp_word))

                if len(ent_list) != len(each_ent_list):
                    print(case_idx)
                    print("tmp_word, tmp_tag: ", tmp_word, tmp_tag, '\n')
                    print("each_sen: ", each_sen, '\n')
                    print(len(ent_list), len(each_ent_list), '\n')
                    print("ent_list: ", ent_list, '\n')
                    print("each_ent_list: ", each_ent_list, '\n')

                assert len(ent_list) == len(each_ent_list)

                if ent_list != ent_str_org_list:
                    print("ent_list: ", ent_list)
                    print("ent_str_org_list: ", ent_str_org_list)
                    print(list([item1 == item2 for item1, item2 in zip(ent_list, ent_str_org_list)]))

                assert ent_list == ent_str_org_list

                if ent_type_org_list != ent_type_list:
                    print(each_sen.split(' '))
                    print(tmp_word, tmp_tag)
                    print("ent_type_org_list: ", ent_type_org_list)
                    print("ent_type_list: ", ent_type_list)

                assert ent_type_org_list == ent_type_list

                ent_id_str_idx_type_list.append(list(zip(ent_id_list, ent_list,
                                                         ent_idx_list, ent_type_list,
                                                         [case_idx] * len(ent_id_list))))

            # Build the id to entity dict for each file
            ent_id_str_idx_type_dict = dict([(item1[0], item1) for item in ent_id_str_idx_type_list for item1 in item])

            ent2senid = dict(
                [(item1[0], item1[-1]) for idx, item in enumerate(ent_id_str_idx_type_list) for item1 in item])

            senid2ent = defaultdict(list)
            for _, item in enumerate(ent_id_str_idx_type_list):
                for item1 in item:
                    senid2ent[item1[-1]].append(ent_id_str_idx_type_dict[item1[0]])

            # Make sure each id links to distinct entity
            if len(set([(item1[0], idx) for idx, item in enumerate(ent_id_str_idx_type_list)
                        for item1 in item])) != len(ent2senid):
                print(txt_file)
                print([(item1[0], item1[-1]) for idx, item in enumerate(ent_id_str_idx_type_list) for item1 in item])
                print(ent2senid)
            assert len(set([(item1[0], idx) for idx, item in enumerate(ent_id_str_idx_type_list)
                            for item1 in item])) == len(ent2senid)

            # Get the positive data
            sen_id_rel_tuple_pos_dict = defaultdict(list)

            for arg1_id, rel_str, arg2_id in arg1_rel_arg2_tuple_list:

                if arg1_id not in ent2senid:
                    # if arg1_id in ent_id_str_idx_type_dict:
                    #     print(txt_file)
                    #     print("The entity {} is not in the dictionary.".format(arg1_id))
                    continue

                if arg2_id not in ent2senid:
                    # if arg2_id in ent_id_str_idx_type_dict:
                    #     print(txt_file)
                    #     print("The entity {} is not in the dictionary.".format(arg2_id))
                    continue

                if ent2senid[arg1_id] != ent2senid[arg2_id]:
                    # print(txt_file)
                    # print("Two entities not in the same sentence.", rel_str, arg1_id, arg2_id,
                    #       ent2senid[arg1_id],
                    #       ent2senid[arg2_id], '\n')
                    continue

                assert arg1_id in ent2senid and arg2_id in ent2senid
                assert arg1_id in ent2senid and arg2_id in ent2senid and ent2senid[arg1_id] == ent2senid[arg2_id]

                if ent_id_str_idx_type_dict[arg1_id][3] in NEW_ENT_TYPE or \
                        ent_id_str_idx_type_dict[arg2_id][3] in NEW_ENT_TYPE:
                    continue

                sen_id_rel_tuple_pos_dict[ent2senid[arg2_id]].append((rel_str,
                                                                      ent_id_str_idx_type_dict[arg1_id],
                                                                      ent_id_str_idx_type_dict[arg2_id]))

            # Generate negative pairs
            sen_id_rel_tuple_neg_dict = defaultdict(list)

            for sen_idx in range(select_num):
                # for sen_idx in range(len(sen_list)):

                pos_pair_id_list = [item[1][0] + ' ' + item[2][0] for item in sen_id_rel_tuple_pos_dict[sen_idx]]

                neg_pair_id_list_all = [item1[0] + ' ' + item2[0] for item1 in senid2ent[sen_idx]
                                        for item2 in senid2ent[sen_idx]
                                        if item1 != item2 and
                                        item1[0] + ' ' + item2[0] not in pos_pair_id_list]

                random.seed(1234)
                neg_pair_id_list = neg_pair_id_list_all

                if not neg_pair_id_list:
                    sen_id_rel_tuple_neg_dict[sen_idx] = neg_pair_id_list

                for item in neg_pair_id_list:
                    arg1_id, arg2_id = item.split(' ')

                    assert arg1_id in ent2senid and arg2_id in ent2senid
                    assert arg1_id in ent2senid and arg2_id in ent2senid and ent2senid[arg1_id] == ent2senid[arg2_id]

                    if ent_id_str_idx_type_dict[arg1_id][3] in NEW_ENT_TYPE or \
                            ent_id_str_idx_type_dict[arg2_id][3] in NEW_ENT_TYPE:
                        continue

                    sen_id_rel_tuple_neg_dict[ent2senid[arg2_id]].append(('no_relation',
                                                                          ent_id_str_idx_type_dict[arg1_id],
                                                                          ent_id_str_idx_type_dict[arg2_id]))

            assert len(final_sen_list) == len(sen_id_rel_tuple_pos_dict) and \
                   len(final_sen_list) == len(sen_id_rel_tuple_neg_dict)

            sen_over_files.append(final_sen_list)
            pos_pairs_over_files.append(sen_id_rel_tuple_pos_dict)
            neg_pairs_over_files.append(sen_id_rel_tuple_neg_dict)

        print(f"The number of selected senteces: {sen_count}")
        processed_sen_over_files = []

        for sen_list, pos_pairs_list, neg_pairs_list in zip(sen_over_files, pos_pairs_over_files, neg_pairs_over_files):

            processed_sen_list = []

            for sen_idx, sen_str in enumerate(sen_list):

                word_list = sen_str.split(' ')

                # Process positive data
                assert sorted(set([item[1] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]])) == \
                       sorted(set([item[2] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]]))

                sorted_ent_list = sorted(set([item[1] for item in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]]),
                                         key=lambda x: x[2][0])

                def mask_all_entities(tokens, sorted_entity_list):

                    ent_dict = dict([(item[2][0], item) for item in sorted_entity_list])

                    word_idx = 0

                    tagged_token_list = []

                    while word_idx < len(tokens):

                        if word_idx not in ent_dict:
                            tagged_token_list.append(tokens[word_idx])
                            word_idx += 1
                        else:
                            ent_id, ent_str, (ent_start, ent_end), ent_type, _ = ent_dict[word_idx]

                            if ent_str.strip() != ' '.join(tokens[ent_start:ent_end]).strip():
                                print(ent_str, ' '.join(tokens[ent_start:ent_end]))

                            assert ent_str.strip() == ' '.join(tokens[ent_start:ent_end]).strip()

                            tagged_token_list.append("[arg1-" + ent_type.lower() + "-start]")
                            tagged_token_list.append("[" + ent_id.lower() + "]")
                            tagged_token_list.append(ent_str)
                            tagged_token_list.append("[arg1-" + ent_type.lower() + "-end]")
                            word_idx = ent_end

                    return tagged_token_list

                processed_word_list = mask_all_entities(word_list, sorted_ent_list)

                processed_sen_list.append(
                    f"{tokenizer.cls_token} {' '.join(processed_word_list)} {tokenizer.eos_token}")

            processed_sen_over_files.append(processed_sen_list)

        start_time = time.time()

        # tokenized_texts = [tokenizer.tokenize(sent) for sent in processed_sen_list][:500]
        tokenized_texts = [[tokenizer.tokenize(sent) for sent in processed_sen_list] for processed_sen_list in
                           processed_sen_over_files]

        print("--- %s seconds ---" % (time.time() - start_time))

        # Remove unrelated entities from tokenzied text
        purified_texts = []
        relation_list = []
        arg1_ent_start_list = []
        arg2_ent_start_list = []

        for tokenized_sen_list, pos_pairs_list, neg_pairs_list in zip(tokenized_texts, pos_pairs_over_files,
                                                                      neg_pairs_over_files):

            for sen_idx, word_list in enumerate(tokenized_sen_list):

                assert len(tokenized_sen_list) == len(pos_pairs_list) and len(tokenized_sen_list) == len(neg_pairs_list)

                # Process positive data
                for relation, arg1, arg2 in pos_pairs_list[sen_idx] + neg_pairs_list[sen_idx]:

                    # Note that this is for speeding up the fa_me_re experiments
                    if relation == "no_relation" and down_sample and random.random() > down_sample_rate:
                        continue

                    arg1_id, arg1_str, arg1_offset, arg1_type, arg1_senid = arg1
                    arg2_id, arg2_str, arg2_offset, arg2_type, arg2_senid = arg2

                    arg1_id = arg1_id.lower()
                    arg2_id = arg2_id.lower()

                    if "[{}]".format(arg1_id) not in word_list or "[{}]".format(arg2_id) not in word_list:
                        print("[{}]".format(arg1_id))
                        print("[{}]".format(arg2_id))
                        print(word_list)

                    assert "[{}]".format(arg1_id) in word_list
                    assert "[{}]".format(arg2_id) in word_list

                    arg1_ent_start = "[arg1-" + arg1_type + "-start]"
                    arg1_ent_end = "[arg1-" + arg1_type + "-end]"

                    arg2_ent_start = "[arg2-" + arg2_type + "-start]"
                    arg2_ent_end = "[arg2-" + arg2_type + "-end]"

                    arg1_ent_start = arg1_ent_start.lower()
                    arg2_ent_start = arg2_ent_start.lower()

                    arg1_ent_start_list.append(arg1_ent_start)
                    arg2_ent_start_list.append(arg2_ent_start)

                    def purify_word_list(tokens, arg1id, arg2id):

                        word_idx = 0
                        arg_tag = 0
                        purified_token_list = []

                        while word_idx < len(tokens):

                            if tokens[word_idx][-len('-start]'):] == '-start]':
                                if tokens[word_idx + 1] == '[' + arg1id + ']':
                                    purified_token_list.append(tokens[word_idx])
                                    arg_tag = 1
                                elif tokens[word_idx + 1] == '[' + arg2id + ']':
                                    purified_token_list.append(
                                        tokens[word_idx][:len('[arg')] + '2' + tokens[word_idx][len('[arg1'):])
                                    arg_tag = 2
                                word_idx += 2
                            elif tokens[word_idx][-len('-end]'):] == '-end]':
                                if arg_tag:
                                    purified_token_list.append(
                                        tokens[word_idx][:len('[arg')] + str(arg_tag) + tokens[word_idx][len('[arg1'):])
                                    arg_tag = 0
                                word_idx += 1
                            else:
                                purified_token_list.append(tokens[word_idx])
                                word_idx += 1

                        return purified_token_list

                    purified_word_list = purify_word_list(word_list, arg1_id, arg2_id)
                    purified_texts.append(purified_word_list)

                    relation_list.append(relation.rstrip(string.digits))

        # print(Counter(relation_list).items())

        assert len(purified_texts) == len(arg1_ent_start_list) and len(purified_texts) == len(arg2_ent_start_list)

        return purified_texts, arg1_ent_start_list, arg2_ent_start_list, relation_list

    if src_data:
        src_data_dir = DATA_DIR[src_data][data_class]
        src_file_order = FILE_ORDER[src_data][data_class]
        purified_texts_src, arg1_ent_start_list_src, arg2_ent_start_list_src, relation_list_src = load_from_dir(src_data_dir, src_file_order)
    else:
        purified_texts_src, arg1_ent_start_list_src, arg2_ent_start_list_src, relation_list_src = [], [], [], []

    if tgt_data:
        tgt_data_dir = DATA_DIR[tgt_data][data_class]
        tgt_file_order = FILE_ORDER[tgt_data][data_class]
        purified_texts_tgt, arg1_ent_start_list_tgt, arg2_ent_start_list_tgt, relation_list_tgt = load_from_dir(tgt_data_dir, tgt_file_order, total_sen_num)
    else:
        purified_texts_tgt, arg1_ent_start_list_tgt, arg2_ent_start_list_tgt, relation_list_tgt = [], [], [], []

    purified_texts = purified_texts_src + purified_texts_tgt
    arg1_ent_start_list = arg1_ent_start_list_src + arg1_ent_start_list_tgt
    arg2_ent_start_list = arg2_ent_start_list_src + arg2_ent_start_list_tgt
    relation_list = relation_list_src + relation_list_tgt
    data_type_list = [[1, 0, alpha]] * len(purified_texts_src) + [[0, 1, 1]] * len(purified_texts_tgt)

    # Get the input_ids and labels
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in purified_texts],
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", truncating="post",
                              padding="post")

    # attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    labels = [rel2idx[l] for l in relation_list]

    # print(purified_texts[0])
    # print(arg1_ent_start_list[0])
    # print(arg2_ent_start_list[0])

    arg1_idx_list = [purified_texts[sen_idx].index(arg1_ent_start_list[sen_idx])
                     if purified_texts[sen_idx].index(arg1_ent_start_list[sen_idx]) < max_len else max_len - 1
                     for sen_idx in range(len(purified_texts))
                     ]

    arg2_idx_list = [purified_texts[sen_idx].index(arg2_ent_start_list[sen_idx])
                     if purified_texts[sen_idx].index(arg2_ent_start_list[sen_idx]) < max_len else max_len - 1
                     for sen_idx in range(len(purified_texts))
                     ]

    # print(len(input_ids), len(attention_masks), len(labels), len(arg1_idx_list), len(arg2_idx_list),
    #       len(data_type_list))

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)
    arg1_idx_list = torch.tensor(arg1_idx_list)
    arg2_idx_list = torch.tensor(arg2_idx_list)
    data_type_list = torch.tensor(data_type_list)

    final_data = TensorDataset(input_ids, attention_masks, labels, arg1_idx_list, arg2_idx_list, data_type_list)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader