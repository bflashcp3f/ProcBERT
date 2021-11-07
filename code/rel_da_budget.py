
import json
import os
import numpy as np
import random
import torch

from datetime import datetime

from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from utils.rel import *

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
        raise ValueError(
            f"The total budget {args.budget} is less than or equal to the pre-training cost {pretraining_cost}.")
    else:
        annotation_cost = args.budget - pretraining_cost
        training_set_cost = TRAINING_SET_ANNOTATION_COST[args.tgt_data]
        if annotation_cost > training_set_cost:
            raise ValueError(
                f"The annotation budget {annotation_cost} is greater than the annotation cost of the {args.tgt_data} training set {training_set_cost}.")

    rel2idx, idx2rel = REL2IDX['wlp'], REL_NAME['wlp']

    args.n_labels = len(rel2idx)
    tokenizer, model, model_name, saved_model_dir = get_model(args)

    # Add new tokens to the vocabulary
    tokenizer.add_tokens(ENT_END['wlp'])
    tokenizer.add_tokens(ENT_START['wlp'])
    tokenizer.add_tokens(ENT_ID)

    # Resize the model
    model.resize_token_embeddings(len(tokenizer))

    # Check whether the experiment has been done
    config_path = os.path.join(saved_model_dir, 'self_config.json')
    if os.path.isfile(config_path):

        with open(config_path) as infile:
            result_dict = json.load(infile)

        if result_dict['epochs'] - 1 == result_dict['current_epoch']:
            return

    train_dataloader = get_processed_sentences_da_budget([args.src_data, args.tgt_data], 'train',
                                                         annotation_cost, args.max_len, args.batch_size,
                                                         tokenizer, args.alpha, args.down_sample, args.down_sample_rate)

    dev_dataloader = get_processed_sentences_da_budget([None, args.tgt_data], 'dev',
                                                       None, args.max_len, args.batch_size,
                                                       tokenizer, args.alpha)

    test_dataloader = get_processed_sentences_da_budget([None, args.tgt_data], 'test',
                                                        None, args.max_len, args.batch_size,
                                                        tokenizer, args.alpha)

    if n_gpu > 1:
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    best_f1 = 0
    config = vars(args).copy()
    config['saved_model_dir'] = saved_model_dir

    # trange is a tqdm wrapper around the normal python range
    for num_epoch in trange(args.epochs, desc="Epoch"):

        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):

            if step % 100 == 0 and step > 0:
                print("The number of steps: {}".format(step), datetime.now())

                if step > 1000:
                    break

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx, b_data_types = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,
                            subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx, data_types=b_data_types
                            )
            if n_gpu > 1:
                loss = loss.mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # dev

        # Put model in evaluation mode to evaluate loss on the dev set
        model.eval()

        dev_gold = []
        dev_pred = []

        # Evaluate data for one epoch
        for batch_index, batch in enumerate(dev_dataloader):

            # print(f"batch_index: {batch_index}")

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx, b_data_types = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up dev
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                               subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx, data_types=b_data_types
                               )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            dev_pred += np.argmax(logits, axis=1).tolist()
            dev_gold += label_ids.tolist()

        print("The performance on dev set: ")
        prec_dev, rec_dev, f1_dev = score([idx2rel[gold_id] for gold_id in dev_gold],
                                          [idx2rel[pred_id] for pred_id in dev_pred],
                                          verbose=False
                                          )
        print("Precision-Score: {}".format(prec_dev))
        print("Recall-Score: {}".format(rec_dev))
        print("F1-Score: {}".format(f1_dev))

        test_gold = []
        test_pred = []

        # Evaluate data for one epoch
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx, b_data_types = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up test
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                               subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx, data_types=b_data_types
                               )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            test_pred += np.argmax(logits, axis=1).tolist()
            test_gold += label_ids.tolist()

        print("\nThe performance on test set: ")
        prec_test, rec_test, f1_test = score([idx2rel[gold_id] for gold_id in test_gold],
                                             [idx2rel[pred_id] for pred_id in test_pred],
                                             verbose=False
                                             )
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
