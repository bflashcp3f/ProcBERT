
import os
import glob
import torch

from utils.utils import *
from chemdataextractor.nlp.tokenize import ChemWordTokenizer

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

CheMU_ENT_NAME = ['EXAMPLE_LABEL', 'OTHER_COMPOUND', 'REACTION_PRODUCT', 'REAGENT_CATALYST', 'SOLVENT', 'STARTING_MATERIAL',
                  'TEMPERATURE', 'TIME', 'YIELD_OTHER', 'YIELD_PERCENT']

USE_HEAD_ONLY = True


def generate_tag2id(ent_name_list):

    tag_name_list = ["O"]

    for each_ent_name in ent_name_list:
        tag_name_list.append('B-' + each_ent_name)
        tag_name_list.append('I-' + each_ent_name)

    tag2id = dict([(value, key) for key, value in enumerate(tag_name_list)])

    return tag_name_list, tag2id


WLP_TAG_NAME, WLP_TAG2IDX = generate_tag2id(WLP_ENT_NAME)
PubMed_TAG_NAME, PubMed_TAG2IDX = generate_tag2id(PubMed_ENT_NAME)
ChemSyn_TAG_NAME, ChemSyn_TAG2IDX = generate_tag2id(ChemSyn_ENT_NAME)
CheMU_TAG_NAME, CheMU_TAG2IDX = generate_tag2id(CheMU_ENT_NAME)

ENT_NAME = {
    'wlp': WLP_ENT_NAME,
    'pubmed': PubMed_ENT_NAME,
    'chemsyn': ChemSyn_ENT_NAME,
    'chemu': CheMU_ENT_NAME,
}

TAG_NAME = {
    'wlp': WLP_TAG_NAME,
    'pubmed': PubMed_TAG_NAME,
    'chemsyn': ChemSyn_TAG_NAME,
    'chemu': CheMU_TAG_NAME,
}

TAG2IDX = {
    'wlp': WLP_TAG2IDX,
    'pubmed': PubMed_TAG2IDX,
    'chemsyn': ChemSyn_TAG2IDX,
    'chemu': CheMU_TAG2IDX,
}


def load_from_brat(txt_file, ent_name_list):

    # print(txt_file)
    ann_file = txt_file[:-3] + "ann"

    if not os.path.isfile(txt_file):
        print(f"Not exists: {txt_file}")
        raise
    elif not os.path.isfile(ann_file):
        print(f"Not exists: {ann_file}")
        raise

    sen_list = load_from_txt(txt_file, strip=False)
    sen_len_list = [len(item) for item in sen_list]

    ann_list = load_from_txt(ann_file)
    all_sen_str = ''.join(sen_list)

    ent_start_list = []

    # Parse ann data
    for item in ann_list:
        if item[0] == 'T':
            try:
                ent_id, label_offset, ent_str = item.split('\t')
            except:
                # print('item split problem')
                # print(ann_file)
                # print(item)
                pass

            try:
                if ';' not in label_offset:
                    ent_label, ent_start, ent_end = label_offset.split(' ')
                    ent_start, ent_end = int(ent_start), int(ent_end)

                    if ent_label not in ent_name_list:
                        continue
                else:
                    continue
            except:
                # print('label_offset split problem')
                # print(label_offset)
                pass

            assert ent_str == all_sen_str[ent_start:ent_end] or \
                   ent_str == all_sen_str[ent_start:ent_end].strip()

            ent_start_list.append((ent_start, (ent_str, ent_start, ent_end, ent_label)))

    # Split entities by sentence
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

            ent_str, ent_start, ent_end, ent_label = ent_info

            # Remove the sentence offset
            ent_start, ent_end = ent_start - sen_start_list[sen_idx], \
                                 ent_end - sen_start_list[sen_idx]

            sen_ent_dict[sen_idx].append((ent_str, ent_start, ent_end, ent_label))

            ent_idx += 1
            continue

        elif ent_start >= sen_start_list[sen_idx + 1]:
            sen_idx += 1
        else:
            print("Bug here")

    conll_list_all = []

    # Generate conll data for each sentence
    for sen_idx in range(len(sen_list)):

        sen_str = sen_list[sen_idx]

        ent_list = sen_ent_dict.get(sen_idx, [])

        sen_str = sen_str.rstrip('\n')
        # print(sen_str)

        # # Scispacy tokenization
        # doc = nlp(sen_str)
        # tokenized_word_list = [token.text for token in doc if len(token.text.split(" ")) <= 1]
        # REMOVED_CHAR = [' ']

        # ChemDataExtractor tokenization
        cwt = ChemWordTokenizer()
        tokenized_word_list = cwt.tokenize(sen_str)
        REMOVED_CHAR = [' ', '\u2009', '\xa0', '\u202f', '\t']

        nonspace_tokenized_sen_str = "".join(tokenized_word_list)

        if "".join(tokenized_word_list) != "".join([item for item in sen_str if item not in REMOVED_CHAR]):
            print(len("".join(tokenized_word_list)),
                  "".join(tokenized_word_list))
            print(len("".join([item for item in sen_str if item not in REMOVED_CHAR])),
                  "".join([item for item in sen_str if item not in REMOVED_CHAR]))

            print([item for item in "".join(tokenized_word_list)])
            print([item for item in sen_str if item not in REMOVED_CHAR])

        assert "".join(tokenized_word_list) == "".join([item for item in sen_str if item not in REMOVED_CHAR])

        # Build the mapping between the original sentence to the tokenized sentence
        # the key is the character index in the original sentence
        # the value is the word index in the tokenized sentence

        tokenized_char2word_idx_dict = {}
        char_offset = 0

        for word_idx, word in enumerate(tokenized_word_list):
            tokenized_char2word_idx_dict.update(dict([(char_idx + char_offset, word_idx)
                                                      for char_idx in range(len(word))]))
            char_offset += len(word)

        assert len(tokenized_char2word_idx_dict) == len(nonspace_tokenized_sen_str)

        org_char_idx = 0
        tokenized_char_idx = 0
        org_char2tokenized_word_idx_dict = {}
        tokenized_word2org_char_idx_dict = defaultdict(list)

        while org_char_idx < len(sen_str) and tokenized_char_idx < len(tokenized_char2word_idx_dict):

            char = sen_str[org_char_idx]

            if char in REMOVED_CHAR:
                org_char_idx += 1
                continue
            else:

                if sen_str[org_char_idx] != nonspace_tokenized_sen_str[tokenized_char_idx]:
                    print(sen_str, org_char_idx)
                    print(nonspace_tokenized_sen_str, tokenized_char_idx)
                    print(sen_str[org_char_idx], nonspace_tokenized_sen_str[tokenized_char_idx])

                assert sen_str[org_char_idx] == nonspace_tokenized_sen_str[tokenized_char_idx]
                tokenized_word_idx = tokenized_char2word_idx_dict[tokenized_char_idx]
                org_char2tokenized_word_idx_dict[org_char_idx] = tokenized_word_idx
                tokenized_word2org_char_idx_dict[tokenized_word_idx].append(org_char_idx)
                org_char_idx += 1
                tokenized_char_idx += 1

        tokenized_ent_list = []

        for ent_str, ent_start, ent_end, ent_label in ent_list:

            # print(ent_str, ent_start, ent_end, ent_label)

            # entity span does not include the last char
            if ent_str != sen_str[ent_start:ent_end] and \
               ent_str.strip() != sen_str[ent_start:ent_end].strip():
                print(sen_str)
                print(ent_list)
                print(ent_str)
                print([i for i in ent_str])
                print(sen_str[ent_start:ent_end])
                print([i for i in sen_str[ent_start:ent_end]])

            assert ent_str == sen_str[ent_start:ent_end] or ent_str.strip() == sen_str[ent_start:ent_end].strip()

            assert ent_start in org_char2tokenized_word_idx_dict

            if sen_str[ent_end - 1] in REMOVED_CHAR:

                while ent_end > ent_start and sen_str[ent_end - 1] in REMOVED_CHAR:
                    ent_end -= 1

            assert ent_end - 1 in org_char2tokenized_word_idx_dict

            ent_start_word_idx = org_char2tokenized_word_idx_dict[ent_start]
            ent_end_word_idx = org_char2tokenized_word_idx_dict[ent_end - 1]

            tokenized_ent_list.append([ent_start_word_idx,
                                       ent_end_word_idx,
                                       ent_start,
                                       ent_end - 1,
                                       tokenized_word_list[ent_start_word_idx:ent_end_word_idx + 1],
                                       ent_label
                                       ])

        tokenized_ent_list_sorted = sorted(tokenized_ent_list, key=lambda x: x[0])

        # Add entity id
        tokenized_ent_list_sorted_dict = dict(
            [(item_idx, item) for item_idx, item in enumerate(tokenized_ent_list_sorted)])

        # Find overlapping entities and resolve the conflit
        def resolve_overlapped_ent_list(ent_dict):

            word_idx2ent_dict = defaultdict(list)

            for entid in sorted(ent_dict.keys()):

                each_ent = ent_dict[entid]
                ent_start_word_idx, ent_end_word_idx, \
                ent_start_org_char_idx, ent_end_org_char_idx, _, _ = each_ent

                for word_idx in range(ent_start_word_idx, ent_end_word_idx + 1):
                    word_idx2ent_dict[word_idx].append(entid)

            removed_entids_all = []

            for word_idx in word_idx2ent_dict.keys():

                if len(word_idx2ent_dict[word_idx]) > 1:

                    def find_entities_to_keep_or_remove(entid_selected, ent_dict_all):

                        ent_len_list = []

                        for entid in entid_selected:
                            ent_start_word_idx, ent_end_word_idx, \
                            ent_start_org_char_idx, ent_end_org_char_idx, _, _ = ent_dict_all[entid]

                            ent_len = ent_end_org_char_idx - ent_start_org_char_idx
                            ent_len_list.append((ent_len, entid, ent_dict_all[entid]))

                        ent_len_list_sorted = sorted(ent_len_list, key=lambda x: x[0], reverse=True)
                        #                             print(ent_len_list_sorted)

                        keep_entid_list = [ent_len_list_sorted[0][1]]
                        remove_entid_list = [item[1] for item in ent_len_list_sorted[1:]]

                        #                             print(keep_entid_list)
                        #                             print(remove_entid_list)

                        return keep_entid_list, remove_entid_list

                    keep_entids, remove_entids = find_entities_to_keep_or_remove(word_idx2ent_dict[word_idx],
                                                                                 ent_dict)
                    removed_entids_all += remove_entids

            # if removed_entids_all:
            #     print([ent for entid, ent in ent_dict.items() if entid in removed_entids_all])

            return [ent for entid, ent in ent_dict.items() if entid not in removed_entids_all]

        tokenized_ent_list_overlap_resolved = resolve_overlapped_ent_list(tokenized_ent_list_sorted_dict)
        # print(tokenized_ent_list_overlap_resolved)
        # print(tokenized_word_list)

        def generate_conll_data(token_list, resolved_ent_list):

            label_list = []
            word_idx = 0

            for each_ent in resolved_ent_list:

                ent_start_word_idx, ent_end_word_idx, \
                ent_start_org_char_idx, ent_end_org_char_idx, \
                ent_word_list, ent_label = each_ent

                if word_idx < ent_start_word_idx:
                    label_list += ['O'] * (ent_start_word_idx - word_idx)

                conll_labels = ([f"B-{ent_label}"] + [f"I-{ent_label}"] * (ent_end_word_idx - ent_start_word_idx))
                label_list += conll_labels

                word_idx = ent_end_word_idx + 1

            if word_idx < len(token_list):
                label_list += ['O'] * (len(token_list) - word_idx)

            assert len(label_list) == len(token_list)

            return [f"{token}\t{label}" for token, label in zip(token_list, label_list)]

        token_label_list = generate_conll_data(tokenized_word_list, tokenized_ent_list_overlap_resolved)
        # print(token_label_list)

        conll_list_all.append(token_label_list)

    return conll_list_all


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


def get_processed_sentences(data_name, data_class, max_len, batch_size, tokenizer):

    data_dir = DATA_DIR[data_name][data_class]
    # file_order = FILE_ORDER[data_name][data_class]
    ent_name = ENT_NAME[data_name]
    tag2idx = TAG2IDX[data_name]

    tokenized_word_list = []
    tokenized_label_list = []
    head_label_list = []
    head_flag_list = []

    print(f"Load data from: {data_dir}")
    txt_file_list = sorted(glob.glob(f'{data_dir}/*.txt'))

    for txt_file in txt_file_list:

        if not os.path.isfile(txt_file):
            print(txt_file)
            continue

        conll_list = load_from_brat(txt_file, ent_name)

        for each_sen in conll_list:

            word_list = [item.split('\t')[0] for item in each_sen]
            label_list = [item.split('\t')[1] for item in each_sen]

            assert len(word_list) == len(label_list)

            def get_subtoken_label(word_list, label_list):

                piece_list_all = []
                flag_list_all = []
                head_label_list_all = []
                piece_label_list_all = []

                for word, word_label in zip(word_list, label_list):

                    if (word_label.startswith('B-') or word_label.startswith('I-')) and word_label[2:] in NEW_ENT_TYPE:
                        # print(word, word_label)
                        word_label = "O"

                    piece_list = tokenizer.tokenize(word)
                    piece_label_list = [word_label] + [word_label.replace("B-", "I-")] * (
                                len(piece_list) - 1) if word_label.startswith("B-") else \
                        [word_label] * len(piece_list)
                    flag_list = [1] + [0] * (len(piece_list) - 1)
                    head_label_list = [word_label] + ["O"] * (len(piece_list) - 1)

                    piece_list_all += piece_list
                    piece_label_list_all += piece_label_list

                    flag_list_all += flag_list
                    head_label_list_all += head_label_list

                assert len(word_list) == sum(flag_list_all)
                assert len(flag_list_all) == len(head_label_list_all)
                assert len(piece_list_all) == len(flag_list_all)
                assert len(piece_list_all) == len(piece_label_list_all)

                # Add "cls" and "eos" for RobertaTokenizer
                piece_list_all = [tokenizer.cls_token] + piece_list_all + [tokenizer.eos_token]

                piece_label_list_all = ["O"] + piece_label_list_all + ["O"]
                head_label_list_all = ["O"] + head_label_list_all + ["O"]
                flag_list_all = [0] + flag_list_all + [0]

                assert len(flag_list_all) == len(head_label_list_all)
                assert len(piece_list_all) == len(flag_list_all)
                assert len(piece_list_all) == len(piece_label_list_all)

                return piece_list_all, piece_label_list_all, head_label_list_all, flag_list_all

            tokenized_word, tokenized_label, head_label, head_flag = get_subtoken_label(word_list, label_list)

            tokenized_word_list.append(tokenized_word)
            tokenized_label_list.append(tokenized_label)
            head_label_list.append(head_label)
            head_flag_list.append(head_flag)

    print(f"The number of sentences: {len(tokenized_word_list)}, the max sentence length: {max([len(item) for item in tokenized_word_list])}")

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_word_list], \
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    tags = pad_sequences([[tag2idx[l] for l in lab] for lab in tokenized_label_list],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    tags = torch.tensor(tags)

    head_tags = pad_sequences([[tag2idx[l] for l in lab] for lab in head_label_list],
                              maxlen=max_len, value=tag2idx["O"], padding="post",
                              dtype="long", truncating="post")
    head_tags = torch.tensor(head_tags)

    head_flags = pad_sequences(head_flag_list,
                               maxlen=max_len, value=0, padding="post",
                               dtype="long", truncating="post")
    head_flags = torch.tensor(head_flags)

    final_data = TensorDataset(inputs, masks, tags, head_tags, head_flags)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader, masks, tags, head_tags, head_flags


def get_processed_sentences_budget(data_name, data_class, budget, max_len, batch_size, tokenizer):

    data_dir = DATA_DIR[data_name][data_class]
    file_order = FILE_ORDER[data_name][data_class]
    total_sen_num = int(budget/DATA_PRICE[data_name]) if budget else None
    ent_name = ENT_NAME['wlp']
    tag2idx = TAG2IDX['wlp']

    tokenized_word_list = []
    tokenized_label_list = []
    head_label_list = []
    head_flag_list = []

    print(f"Load data from: {data_dir}")

    sen_count = 0

    txt_file_list = load_from_txt(file_order)

    for txt_file in txt_file_list:

        if not os.path.isfile(f"{data_dir}/{txt_file}"):
            print(f"{data_dir}/{txt_file}")
            continue

        conll_list = load_from_brat(f"{data_dir}/{txt_file}", ent_name)

        if total_sen_num:

            if sen_count >= total_sen_num:
                break
            else:
                if sen_count + len(conll_list) <= total_sen_num:
                    select_num = len(conll_list)
                else:
                    select_num = total_sen_num - sen_count

                sen_count += select_num
        else:
            select_num = len(conll_list)

        for each_sen in conll_list[:select_num]:

            word_list = [item.split('\t')[0] for item in each_sen]
            label_list = [item.split('\t')[1] for item in each_sen]

            assert len(word_list) == len(label_list)

            def get_subtoken_label(word_list, label_list):

                piece_list_all = []
                flag_list_all = []
                head_label_list_all = []
                piece_label_list_all = []

                for word, word_label in zip(word_list, label_list):

                    if (word_label.startswith('B-') or word_label.startswith('I-')) and word_label[2:] in NEW_ENT_TYPE:
                        # print(word, word_label)
                        word_label = "O"

                    piece_list = tokenizer.tokenize(word)
                    piece_label_list = [word_label] + [word_label.replace("B-", "I-")] * (
                                len(piece_list) - 1) if word_label.startswith("B-") else \
                        [word_label] * len(piece_list)
                    flag_list = [1] + [0] * (len(piece_list) - 1)
                    head_label_list = [word_label] + ["O"] * (len(piece_list) - 1)

                    piece_list_all += piece_list
                    piece_label_list_all += piece_label_list

                    flag_list_all += flag_list
                    head_label_list_all += head_label_list

                assert len(word_list) == sum(flag_list_all)
                assert len(flag_list_all) == len(head_label_list_all)
                assert len(piece_list_all) == len(flag_list_all)
                assert len(piece_list_all) == len(piece_label_list_all)

                # Add "cls" and "eos" for RobertaTokenizer
                piece_list_all = [tokenizer.cls_token] + piece_list_all + [tokenizer.eos_token]

                piece_label_list_all = ["O"] + piece_label_list_all + ["O"]
                head_label_list_all = ["O"] + head_label_list_all + ["O"]
                flag_list_all = [0] + flag_list_all + [0]

                assert len(flag_list_all) == len(head_label_list_all)
                assert len(piece_list_all) == len(flag_list_all)
                assert len(piece_list_all) == len(piece_label_list_all)

                return piece_list_all, piece_label_list_all, head_label_list_all, flag_list_all

            tokenized_word, tokenized_label, head_label, head_flag = get_subtoken_label(word_list, label_list)

            tokenized_word_list.append(tokenized_word)
            tokenized_label_list.append(tokenized_label)
            head_label_list.append(head_label)
            head_flag_list.append(head_flag)

    print(f"The number of sentences: {len(tokenized_word_list)}, the max sentence length: {max([len(item) for item in tokenized_word_list])}")

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_word_list], \
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    tags = pad_sequences([[tag2idx[l] for l in lab] for lab in tokenized_label_list],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    tags = torch.tensor(tags)

    head_tags = pad_sequences([[tag2idx[l] for l in lab] for lab in head_label_list],
                              maxlen=max_len, value=tag2idx["O"], padding="post",
                              dtype="long", truncating="post")
    head_tags = torch.tensor(head_tags)

    head_flags = pad_sequences(head_flag_list,
                               maxlen=max_len, value=0, padding="post",
                               dtype="long", truncating="post")
    head_flags = torch.tensor(head_flags)

    final_data = TensorDataset(inputs, masks, tags, head_tags, head_flags)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader, masks, tags, head_tags, head_flags


def get_processed_sentences_da_budget(data_name_list, data_class, budget, max_len, batch_size, tokenizer, alpha=1):

    src_data, tgt_data = data_name_list
    ent_name = ENT_NAME['wlp']
    tag2idx = TAG2IDX['wlp']

    if not tgt_data:
        raise ValueError("There must be a target domain")

    total_sen_num = int(budget / DATA_PRICE[tgt_data]) if budget else None

    if total_sen_num:
        print(f"Select top {total_sen_num} sentences from the target domain.")


    def load_from_dir(data_dir, file_order, total_sen_num=None):

        tokenized_word_list = []
        tokenized_label_list = []
        head_label_list = []
        head_flag_list = []

        print(f"Load data from: {data_dir}")
        sen_count = 0

        txt_file_list = load_from_txt(file_order)

        for txt_file in txt_file_list:

            if not os.path.isfile(f"{data_dir}/{txt_file}"):
                print(f"{data_dir}/{txt_file}")
                continue

            conll_list = load_from_brat(f"{data_dir}/{txt_file}", ent_name)

            if total_sen_num:

                if sen_count >= total_sen_num:
                    break
                else:
                    if sen_count + len(conll_list) <= total_sen_num:
                        select_num = len(conll_list)
                    else:
                        select_num = total_sen_num - sen_count

                    sen_count += select_num
            else:
                select_num = len(conll_list)

            for each_sen in conll_list[:select_num]:

                word_list = [item.split('\t')[0] for item in each_sen]
                label_list = [item.split('\t')[1] for item in each_sen]

                assert len(word_list) == len(word_list)

                def get_subtoken_label(word_list, label_list):

                    piece_list_all = []
                    flag_list_all = []
                    head_label_list_all = []
                    piece_label_list_all = []

                    for word, word_label in zip(word_list, label_list):

                        if (word_label.startswith('B-') or word_label.startswith('I-')) and word_label[2:] in NEW_ENT_TYPE:
                            # print(word, word_label)
                            word_label = "O"

                        piece_list = tokenizer.tokenize(word)
                        piece_label_list = [word_label] + [word_label.replace("B-", "I-")] * (
                                len(piece_list) - 1) if word_label.startswith("B-") else \
                            [word_label] * len(piece_list)
                        flag_list = [1] + [0] * (len(piece_list) - 1)
                        head_label_list = [word_label] + ["O"] * (len(piece_list) - 1)

                        piece_list_all += piece_list
                        piece_label_list_all += piece_label_list

                        flag_list_all += flag_list
                        head_label_list_all += head_label_list

                    assert len(word_list) == sum(flag_list_all)
                    assert len(flag_list_all) == len(head_label_list_all)
                    assert len(piece_list_all) == len(flag_list_all)
                    assert len(piece_list_all) == len(piece_label_list_all)

                    # Add "cls" and "eos" for RobertaTokenizer
                    piece_list_all = [tokenizer.cls_token] + piece_list_all + [tokenizer.eos_token]

                    piece_label_list_all = ["O"] + piece_label_list_all + ["O"]
                    head_label_list_all = ["O"] + head_label_list_all + ["O"]
                    flag_list_all = [0] + flag_list_all + [0]

                    assert len(flag_list_all) == len(head_label_list_all)
                    assert len(piece_list_all) == len(flag_list_all)
                    assert len(piece_list_all) == len(piece_label_list_all)

                    return piece_list_all, piece_label_list_all, head_label_list_all, flag_list_all

                tokenized_word, tokenized_label, head_label, head_flag = get_subtoken_label(word_list, label_list)

                tokenized_word_list.append(tokenized_word)
                tokenized_label_list.append(tokenized_label)
                head_label_list.append(head_label)
                head_flag_list.append(head_flag)

        print(len(tokenized_word_list), max([len(item) for item in tokenized_word_list]))

        return tokenized_word_list, tokenized_label_list, head_label_list, head_flag_list

    # Source data
    if src_data:
        src_data_dir, src_file_order = DATA_DIR[src_data][data_class], FILE_ORDER[src_data][data_class]
        tokenized_word_list_src, tokenized_label_list_src, \
        head_label_list_src, head_flag_list_src = load_from_dir(src_data_dir, src_file_order)
    else:
        tokenized_word_list_src = []
        tokenized_label_list_src = []
        head_label_list_src = []
        head_flag_list_src = []

    # Target data
    if data_class != "train" or (data_class == "train" and total_sen_num and total_sen_num > 0):
        tgt_data_dir, tgt_file_order = DATA_DIR[tgt_data][data_class], FILE_ORDER[tgt_data][data_class]
        tokenized_word_list_tgt, tokenized_label_list_tgt, \
        head_label_list_tgt, head_flag_list_tgt = load_from_dir(tgt_data_dir, tgt_file_order, total_sen_num)
    else:
        tokenized_word_list_tgt = []
        tokenized_label_list_tgt = []
        head_label_list_tgt = []
        head_flag_list_tgt = []

    tokenized_word_list = tokenized_word_list_src + tokenized_word_list_tgt
    tokenized_label_list = tokenized_label_list_src + tokenized_label_list_tgt
    head_label_list = head_label_list_src + head_label_list_tgt
    head_flag_list = head_flag_list_src + head_flag_list_tgt
    data_type_list = [[1, 0, alpha]] * len(tokenized_word_list_src) + [[0, 1, 1]] * len(tokenized_word_list_tgt)

    assert len(tokenized_word_list) == len(data_type_list)
    assert len(tokenized_word_list) == len(tokenized_label_list)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_word_list], \
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    tags = pad_sequences([[tag2idx[l] for l in lab] for lab in tokenized_label_list],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    tags = torch.tensor(tags)

    head_tags = pad_sequences([[tag2idx[l] for l in lab] for lab in head_label_list],
                              maxlen=max_len, value=tag2idx["O"], padding="post",
                              dtype="long", truncating="post")
    head_tags = torch.tensor(head_tags)

    head_flags = pad_sequences(head_flag_list,
                               maxlen=max_len, value=0, padding="post",
                               dtype="long", truncating="post")
    head_flags = torch.tensor(head_flags)

    data_types = torch.tensor(data_type_list)

    final_data = TensorDataset(inputs, masks, tags, head_tags, head_flags, data_types)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader, masks, tags, head_tags, head_flags, data_types


def get_processed_sentences_demo_load_merge(data_name, doc_sens, max_len, batch_size, data_class, tokenizer, idx_start, idx_end):
    """
    Consider the head only cases
    :param doc_sens:
    :param max_len:
    :param batch_size:
    :param data_class:
    :param tokenizer:
    :return:
    """

    tag2idx = TAG2IDX[data_name]

    tokenized_word_list = []
    untokenized_word_list = []
    tokenized_label_list = []
    head_label_list = []
    head_flag_list = []
    file_id_list = []
    long_flag_list = []

    # ChemDataExtractor tokenization
    cwt = ChemWordTokenizer()

    for txt_file in sorted(doc_sens.keys())[idx_start:idx_end]:

        sen_list = doc_sens[txt_file]

        for each_sen in sen_list:

            # word_list = [item for item in word_tokenize(each_sen) if item]
            word_list = cwt.tokenize(each_sen)

            label_list = ['O'] * len(word_list)

            assert len(word_list) == len(word_list)

            def get_subtoken_label(word_list, label_list):

                piece_list_all = []
                flag_list_all = []
                head_label_list_all = []
                piece_label_list_all = []

                for word, word_label in zip(word_list, label_list):

                    piece_list = tokenizer.tokenize(word)

                    # Some tokens are missed after bert tokenization like '̊ '
                    if not piece_list:
                        piece_list = [tokenizer.unk_token]

                    piece_label_list = [word_label] + [word_label.replace("B-", "I-")] * (len(piece_list) - 1) \
                        if word_label.startswith("B-") else [word_label] * len(piece_list)
                    flag_list = [1] + [0] * (len(piece_list) - 1)
                    head_label_list = [word_label] + ["O"] * (len(piece_list) - 1)

                    piece_list_all += piece_list
                    piece_label_list_all += piece_label_list

                    flag_list_all += flag_list
                    head_label_list_all += head_label_list

                #                     print(piece_list)
                #                     print(flag_list)
                #                     print(word_label)
                #                     print()
                #                 print(piece_list_all)
                #                 print(piece_label_list_all)
                #                 print(head_label_list_all)
                #                 print(flag_list_all)
                assert len(word_list) == sum(flag_list_all)
                assert len(flag_list_all) == len(head_label_list_all)

                assert len(piece_list_all) == len(flag_list_all)
                assert len(piece_list_all) == len(piece_label_list_all)

                # Add "cls" and "eos" for RobertaTokenizer
                piece_list_all = [tokenizer.cls_token] + piece_list_all + [tokenizer.eos_token]

                piece_label_list_all = ["O"] + piece_label_list_all + ["O"]
                head_label_list_all = ["O"] + head_label_list_all + ["O"]
                flag_list_all = [0] + flag_list_all + [0]

                assert len(flag_list_all) == len(head_label_list_all)
                assert len(piece_list_all) == len(flag_list_all)
                assert len(piece_list_all) == len(piece_label_list_all)

                return piece_list_all, piece_label_list_all, head_label_list_all, flag_list_all

            tokenized_word, tokenized_label, head_label, head_flag = get_subtoken_label(word_list, label_list)
            assert sum(head_flag) == len(word_list)

            tokenized_word_list.append(tokenized_word)
            untokenized_word_list.append(word_list)
            tokenized_label_list.append(tokenized_label)
            head_label_list.append(head_label)
            head_flag_list.append(head_flag)
            # file_id_list.append(txt_file.split('/')[-1].split('.')[0])
            file_id_list.append(txt_file)
            long_flag_list.append(len(tokenized_word) > max_len)

    print(len(tokenized_word_list), max([len(item) for item in tokenized_word_list]))

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_word_list], \
                              maxlen=max_len, value=tokenizer.pad_token_id, dtype="long", \
                              truncating="post", padding="post")
    attention_masks = [[float(i != tokenizer.pad_token_id) for i in ii] for ii in input_ids]

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    tags = pad_sequences([[tag2idx[l] for l in lab] for lab in tokenized_label_list],
                         maxlen=max_len, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")
    tags = torch.tensor(tags)

    head_tags = pad_sequences([[tag2idx[l] for l in lab] for lab in head_label_list],
                              maxlen=max_len, value=tag2idx["O"], padding="post",
                              dtype="long", truncating="post")
    head_tags = torch.tensor(head_tags)

    head_flags = pad_sequences(head_flag_list,
                               maxlen=max_len, value=0, padding="post",
                               dtype="long", truncating="post")
    head_flags = torch.tensor(head_flags)

    final_data = TensorDataset(inputs, masks, tags, head_tags, head_flags)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader, masks, tags, head_tags, head_flags, untokenized_word_list, file_id_list, long_flag_list


def get_processed_sentences_t5(data_name, data_class, max_len, batch_size, tokenizer, full_sentence=True):

    data_dir = DATA_DIR[data_name][data_class]
    # file_order = FILE_ORDER[data_name][data_class]
    ent_name = ENT_NAME[data_name]
    tag2idx = TAG2IDX[data_name]

    source_list = []
    target_list = []

    print(f"Load data from: {data_dir}")
    txt_file_list = sorted(glob.glob(f'{data_dir}/*.txt'))

    for txt_file in txt_file_list[:]:

        if not os.path.isfile(txt_file):
            print(txt_file)
            continue

        conll_list = load_from_brat(txt_file, ent_name)

        for each_sen in conll_list:

            word_list = [item.split('\t')[0] for item in each_sen]
            label_list = [item.split('\t')[1] for item in each_sen]

            assert len(word_list) == len(label_list)

            ent_list, ent_idx_list, ent_type_list = index_ent_in_prediction(word_list, label_list)

            def mask_entities(tokens, ent_list, ent_idx_list, ent_type_list, full_sentence=True):

                ent_dict = {item[1][0]:item for item in zip(ent_list, ent_idx_list, ent_type_list)}

                word_idx = 0

                tagged_token_list = []

                while word_idx < len(tokens):

                    if word_idx not in ent_dict:
                        if full_sentence:
                            tagged_token_list.append(tokens[word_idx])
                        word_idx += 1
                    else:
                        ent_str, (ent_start, ent_end), ent_type = ent_dict[word_idx]
                        ent_end += 1

                        if ent_str.strip() != ' '.join(tokens[ent_start:ent_end]).strip():
                            print(ent_str, ' '.join(tokens[ent_start:ent_end]))

                        assert ent_str.strip() == ' '.join(tokens[ent_start:ent_end]).strip()

                        # tagged_token_list.append(f"[{ent_type}-START]")
                        # tagged_token_list.append(ent_str)
                        tagged_token_list.append(f"<{ent_type.lower()}>")
                        tagged_token_list += tokens[ent_start:ent_end]
                        tagged_token_list.append(f"</{ent_type.lower()}>")
                        # tagged_token_list.append(f"[{ent_type}-END]")
                        word_idx = ent_end

                return tagged_token_list

            tagged_token_list = mask_entities(word_list, ent_list, ent_idx_list, ent_type_list, full_sentence=full_sentence)

            # print(word_list)
            # # print(ent_list, ent_idx_list, ent_type_list)
            # print(tokenizer.tokenize(" ".join(word_list)))
            #
            # print(tagged_token_list)
            # print(tokenizer.tokenize(" ".join(tagged_token_list)))
            # print()

            source_list.append(" ".join(word_list))
            target_list.append(" ".join(tagged_token_list))

    # encode the sources
    source_encoding = tokenizer(source_list,
                                padding='longest',
                                max_length=max_len,
                                truncation=True,
                                return_tensors="pt")
    input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(target_list,
                                padding='longest',
                                max_length=max_len,
                                truncation=True)
    labels = target_encoding.input_ids

    # replace padding token id's of the labels by -100
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
    ]
    labels = torch.tensor(labels)

    print(input_ids.size(), attention_mask.size(), labels.size())

    # print(source_list[1])
    # print(tokenizer.tokenize(source_list[1]))
    # print(input_ids[1])
    # print(target_list[1])
    # print(tokenizer.tokenize(target_list[1]))
    # print(labels[1])

    final_data = TensorDataset(input_ids, attention_mask, labels)
    if data_class == "train":
        final_sampler = RandomSampler(final_data)
    else:
        final_sampler = SequentialSampler(final_data)
    final_dataloader = DataLoader(final_data, sampler=final_sampler, batch_size=batch_size)

    return final_dataloader
