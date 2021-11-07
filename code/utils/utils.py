import argparse
import json

from models.ner import *
from models.rel import *
from models.mrc import *
from models.fa_ner import *
from models.fa_rel import *

DATA_PATH = '/srv/share5/fbai31/ProcBERT/anno_data'

PubMed_DIR = {
    'train': f'{DATA_PATH}/pubmed/train/',
    'dev': f'{DATA_PATH}/pubmed/dev/',
    'test': f'{DATA_PATH}/pubmed/test/',
}

ChemSyn_DIR = {
    'train': f'{DATA_PATH}/chemsyn/train/',
    'dev': f'{DATA_PATH}/chemsyn/dev/',
    'test': f'{DATA_PATH}/chemsyn/test/',
}

WLP_DIR = {
    'train': f'{DATA_PATH}/wlp/train/',
    'dev': f'{DATA_PATH}/wlp/dev/',
    'test': f'{DATA_PATH}/wlp/test/',
}

CheMU_DIR = {
    'train': f'{DATA_PATH}/ChEMU/ner/train/',
    'dev': f'{DATA_PATH}/ChEMU/ner/dev/',
    'test': f'{DATA_PATH}/ChEMU/ner/dev/',
}

DATA_DIR = {
    'wlp': WLP_DIR,
    'pubmed': PubMed_DIR,
    'chemsyn': ChemSyn_DIR,
    'chemu': CheMU_DIR,
}

DATA_PRICE = {
    'wlp': 0.44,
    'pubmed': 1.02,
    'chemsyn': 0.60,
}

WLP_ORDER = {
    'train': 'data/file_order/wlp_train.txt',
    'dev': 'data/file_order/wlp_dev.txt',
    'test': 'data/file_order/wlp_test.txt',
}

PubMed_ORDER = {
    'train': 'data/file_order/pubmed_train.txt',
    'dev': 'data/file_order/pubmed_dev.txt',
    'test': 'data/file_order/pubmed_test.txt',
}

ChemSyn_ORDER = {
    'train': 'data/file_order/chemsyn_train.txt',
    'dev': 'data/file_order/chemsyn_dev.txt',
    'test': 'data/file_order/chemsyn_test.txt',
}

FILE_ORDER = {
    'wlp': WLP_ORDER,
    'pubmed': PubMed_ORDER,
    'chemsyn': ChemSyn_ORDER,
}

PRETRAINING_COST = {
    'bert': 0,
    'bertlarge': 0,
    'scibert': 1340,
    'procroberta': 800,
    'procbert': 620,
}

TRAINING_SET_ANNOTATION_COST = {
    'wlp': 5003,
    'pubmed': 880,
    'chemsyn': 4002,
}


def load_from_jsonl(file_name):
    data_list = []
    with open(file_name) as f:
        for line in f:
            data_list.append(json.loads(line))

    return data_list


def load_conll(data_file):

    examples = []
    cache = []
    with open(data_file) as infile:
        while True:
            line = infile.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if len(line) == 0: # empty after strip
                if len(cache) > 0:
                    examples.append(cache)
                    cache = []
            else:
                cache.append(line)
        if len(cache) > 0:
            examples.append(cache)
    return examples


def load_from_txt(data_path, verbose=False, strip=True):
    examples = []

    with open(data_path, "r", encoding='utf-8') as infile:
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


def get_model(args):

    if args.lm_model == "bert":
        model_name = "bert-base-uncased"
    elif args.lm_model == "bertlarge":
        model_name = "bert-large-uncased"
    elif args.lm_model == "roberta":
        model_name = "roberta-base"
    elif args.lm_model == "robertalarge":
        model_name = "roberta-large"
    elif args.lm_model == "biomed":
        model_name = "allenai/biomed_roberta_base"
    elif args.lm_model == 'scibert':
        model_name = "allenai/scibert_scivocab_uncased"
    elif args.lm_model == "procroberta":
        model_name = "fbaigt/proc_roberta"
    elif args.lm_model == "procbert":
        model_name = "fbaigt/procbert"
    else:
        raise

    print("Load the model checkpoint from: ", model_name)

    if args.lm_model in ["bert", "bertlarge", "scibert", "procbert"]:
        tokenizer = BertTokenizer.from_pretrained(model_name)

        config = BertConfig.from_pretrained(model_name)
        config.num_labels = args.n_labels
        if args.task_name == 'ner':
            model = BertForNER.from_pretrained(model_name, config=config)
        elif args.task_name == 'rel':
            model = BertSimpleEMES.from_pretrained(model_name, config=config)
        elif args.task_name == 'mrc':
            model = BertMRC.from_pretrained(model_name, config=config)
        elif args.task_name == 'fa_ner':
            model = BertForNERFA.from_pretrained(model_name, config=config)
        elif args.task_name == 'fa_me_ner':
            model = BertForNERFAMultiEncoder.from_pretrained(model_name, config=config)
            model.init_multi_encoder()
        elif args.task_name == 'fa_rel':
            model = BertSimpleEMESFA.from_pretrained(model_name, config=config)
        else:
            raise

    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        config = RobertaConfig.from_pretrained(model_name)
        config.num_labels = args.n_labels

        if args.task_name == 'ner':
            model = RobertaForNER.from_pretrained(model_name, config=config)
        elif args.task_name == 'rel':
            model = RobertaSimpleEMES.from_pretrained(model_name, config=config)
        elif args.task_name == 'mrc':
            model = RobertaMRC.from_pretrained(model_name, config=config)
        elif args.task_name == 'fa_ner':
            model = RobertaForNERFA.from_pretrained(model_name, config=config)
        elif args.task_name == 'fa_rel':
            model = RobertaSimpleEMESFA.from_pretrained(model_name, config=config)
        else:
            raise

    exp_setup = f'lr_{args.learning_rate}_epoch_{args.epochs}_bs_{args.batch_size}_maxlen_{args.max_len}'

    if args.eval_step:
        exp_setup = f'{exp_setup}_eval_step_{args.eval_step}'

    if args.task_name.startswith('fa'):
        exp_setup = f'{exp_setup}_alpha_{args.alpha}'

    if args.budget > 0:
        exp_setup = f'budget_{args.budget}_{exp_setup}'

    saved_model_dir = f"{args.output_dir}/{args.task_name}/{args.lm_model}/{exp_setup}/seed_{args.random_seed}"

    return tokenizer, model, model_name, saved_model_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--lm_model", default=None, type=str, required=True)
    parser.add_argument('--ckp_num', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_sen', type=int, default=10)
    parser.add_argument('--n_labels', type=int)
    parser.add_argument('--batch_size', type=int, default=16, required=True)
    parser.add_argument('--max_len', type=int, default=256, required=True)
    parser.add_argument('--patient', type=int, default=30)
    parser.add_argument('--eval_step', type=int, default=0)
    parser.add_argument('--budget', type=int, default=0)
    parser.add_argument('--num_subset_file', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--down_sample_rate', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=1234,
                        help="random seed for random library")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--save_model", action='store_true', help="Save trained checkpoints.")
    parser.add_argument("--down_sample", action='store_true', help="Sample the negative data in the training.")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--src_data', type=str, default='')
    parser.add_argument('--tgt_data', type=str, default='')
    parser.add_argument('--data_name', type=str, default='')

    args = parser.parse_args()

    return args

