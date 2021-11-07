

import math
import glob
import time
import json
import pickle
import os
import numpy as np
import operator
import time
import sys
import random

import torch
from transformers import AdamW
from tqdm import tqdm, trange

from utils.ner import *
from utils.conlleval import evaluate

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.random_seed)

    pretraining_cost = PRETRAINING_COST[args.lm_model]
    if args.budget <= pretraining_cost:
        raise ValueError(f"The total budget {args.budget} is less than or equal to the pre-training cost {pretraining_cost}.")
    else:
        annotation_cost = args.budget - pretraining_cost
        training_set_cost = TRAINING_SET_ANNOTATION_COST[args.data_name]
        if annotation_cost > training_set_cost:
            raise ValueError(
                f"The annotation budget {annotation_cost} is greater than the annotation cost of the {args.data_name} training set {training_set_cost}.")

    args.n_labels = len(TAG2IDX['wlp'])
    tag_name = TAG_NAME['wlp']
    tokenizer, model, model_name, saved_model_dir = get_model(args)

    # Check whether the experiment has been done
    config_path = os.path.join(saved_model_dir, 'self_config.json')
    if os.path.isfile(config_path):

        with open(config_path) as infile:
            result_dict = json.load(infile)

        if result_dict['epochs'] - 1 == result_dict['current_epoch']:
            return

    train_dataloader, train_masks, train_tags, \
    train_head_tags, train_head_flags = get_processed_sentences_budget(args.data_name, 'train', annotation_cost,
                                                                       args.max_len, args.batch_size, tokenizer)

    dev_dataloader, dev_masks, dev_tags, \
    dev_head_tags, dev_head_flags = get_processed_sentences_budget(args.data_name, 'dev', None,
                                                                   args.max_len, args.batch_size, tokenizer)

    test_dataloader, test_masks, test_tags, \
    test_head_tags, test_head_flags = get_processed_sentences_budget(args.data_name, 'test', None, args.max_len,
                                                                     args.batch_size, tokenizer)

    # model.cuda();
    if n_gpu > 1:
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = 0
    config = vars(args).copy()
    config['saved_model_dir'] = saved_model_dir

    for num_epoch in trange(args.epochs, desc="Epoch"):

        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):

            if step % 100 == 0 and step > 0:
                print("The number of steps: {}".format(step))

            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_head_tags, b_head_flags = batch
            # forward pass
            if not USE_HEAD_ONLY:
                loss, _ = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            else:
                loss, _ = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels,
                                head_tags=b_head_tags, head_flags=b_head_flags)

            if n_gpu > 1:
                loss = loss.mean()

            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # on dev set
        model.eval()
        predictions, true_labels = [], []
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_head_tags, b_head_flags = batch

            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

        if not USE_HEAD_ONLY:
            pred_tags_str = [tag_name[p_i] for p_idx, p in enumerate(predictions)
                             for p_i_idx, p_i in enumerate(p)
                             if dev_masks[p_idx][p_i_idx]
                             ]
            dev_tags_str = [tag_name[l_i.tolist()] for l_idx, l in enumerate(dev_tags)
                            for l_i_idx, l_i in enumerate(l)
                            if dev_masks[l_idx][l_i_idx]
                            ]
        else:
            pred_tags_str = [tag_name[p_i] for p_idx, p in enumerate(predictions)
                             for p_i_idx, p_i in enumerate(p)
                             if dev_head_flags[p_idx][p_i_idx]
                             ]
            dev_tags_str = [tag_name[l_i.tolist()] for l_idx, l in enumerate(dev_head_tags)
                            for l_i_idx, l_i in enumerate(l)
                            if dev_head_flags[l_idx][l_i_idx]
                            ]

        # https://github.com/sighsmile/conlleval
        prec_dev, rec_dev, f1_dev = evaluate(dev_tags_str, pred_tags_str, verbose=False)
        print("\nOn dev set: ")
        print("Precision-Score: {}".format(prec_dev))
        print("Recall-Score: {}".format(rec_dev))
        print("F1-Score: {}".format(f1_dev))
        print()

        # on test set
        predictions, true_labels = [], []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_head_tags, b_head_flags = batch

            with torch.no_grad():
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

        if not USE_HEAD_ONLY:
            pred_tags_str = [tag_name[p_i] for p_idx, p in enumerate(predictions)
                             for p_i_idx, p_i in enumerate(p)
                             if test_masks[p_idx][p_i_idx]
                             ]
            test_tags_str = [tag_name[l_i.tolist()] for l_idx, l in enumerate(test_tags)
                             for l_i_idx, l_i in enumerate(l)
                             if test_masks[l_idx][l_i_idx]
                             ]
        else:
            pred_tags_str = [tag_name[p_i] for p_idx, p in enumerate(predictions)
                             for p_i_idx, p_i in enumerate(p)
                             if test_head_flags[p_idx][p_i_idx]
                             ]
            test_tags_str = [tag_name[l_i.tolist()] for l_idx, l in enumerate(test_head_tags)
                             for l_i_idx, l_i in enumerate(l)
                             if test_head_flags[l_idx][l_i_idx]
                             ]

        # https://github.com/sighsmile/conlleval
        prec_test, rec_test, f1_test = evaluate(test_tags_str, pred_tags_str, verbose=False)
        print("\nOn test set: ")
        print("Precision-Score: {}".format(prec_test))
        print("Recall-Score: {}".format(rec_test))
        print("F1-Score: {}".format(f1_test))
        print()

        config['current_epoch'] = num_epoch

        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)

        # Check if the current model is the best model
        if f1_dev > best_f1:
            best_f1 = f1_dev

            config['precision_dev'] = prec_dev
            config['recall_dev'] = rec_dev
            config['f1_dev'] = f1_dev

            config['precision_test'] = prec_test
            config['recall_test'] = rec_test
            config['f1_test'] = f1_test

            config['best_epoch'] = num_epoch

            print(config, '\n')

            # # Save the model
            # if args.save_model:
            #
            #     print("\nSave the best model to {}".format(saved_model_dir))
            #     if n_gpu > 1:
            #         model.module.save_pretrained(save_directory=saved_model_dir)
            #     else:
            #         model.save_pretrained(save_directory=saved_model_dir)
            #     tokenizer.save_pretrained(save_directory=saved_model_dir)


        # Save hyper-parameters (lr, batch_size, epoch, precision, recall, f1)
        config_path = os.path.join(saved_model_dir, 'self_config.json')
        with open(config_path, 'w') as json_file:
            json.dump(config, json_file)



if __name__ == '__main__':

    args = get_args()

    for random_seed in [1989894904, 2294922467, 2002866410, 1004506748, 4076792239]:
        args.random_seed = random_seed

        print(args)
        main(args)

