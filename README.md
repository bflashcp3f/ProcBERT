 Pre-train or Annotate? Domain Adaptation with a Constrained Budget
 =========================

This repo contains code and data associated with EMNLP 2021 paper ["Pre-train or Annotate? Domain Adaptation with a Constrained Budget"](https://arxiv.org/abs/2109.04711).
```
@article{Bai2021PretrainOA,
  title={Pre-train or Annotate? Domain Adaptation with a Constrained Budget},
  author={Fan Bai and Alan Ritter and Wei Xu},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.04711}
}
```

## Installment
```
git clone https://github.com/bflashcp3f/ProcBERT.git
cd ProcBERT
conda env create -f environment.yml
conda activate procbert
```

## Data & Model Checkpoints
Three procedural-text datasets (WLP, PubMed and ChemSyn) can be downloaded [here](https://drive.google.com/file/d/1kCtmU3ZfMR2GgLq-NQigdSh5k4IMO9Av/view?usp=sharing), and 
model checkpoints (ProcBERT and Proc-RoBERTa) are accessible through [HuggingFace](https://huggingface.co/fbaigt). 

## Experiment

### Setup
```angular2html
# After downloading the data, update the DATA_PATH variable in code/utils.py
DATA_PATH=<DATA_PATH>

# Set up OUTPUT_DIR before running experiments
export OUTPUT_DIR=<OUTPUT_DIR>
```

### Budget-aware Domain Adaptation Experiments  
```angular2html
# Named Entity Recognition (NER) 
python code/ner_da_budget.py     \
  --lm_model procbert     \
  --src_data pubmed     \
  --tgt_data chemsyn     \
  --gpu_ids 0,1   \
  --output_dir "$OUTPUT_DIR"/da/pubmed_chemsyn     \
  --learning_rate 1e-5     \
  --task_name fa_ner     \
  --batch_size 16     \
  --max_len 512    \
  --epochs 25 \
  --budget 700 \
  --alpha 1   \
  --save_model

# Relation Extraction (RE)
python code/rel_da_budget.py \
  --lm_model procbert \
  --src_data pubmed     \
  --tgt_data chemsyn     \
  --gpu_ids 0,1  \
  --output_dir "$OUTPUT_DIR"/da/pubmed_chemsyn \
  --learning_rate 1e-5 \
  --task_name fa_rel \
  --batch_size 48 \
  --max_len 256 \
  --epochs 5 \
  --budget 700 \
  --alpha 1 \
  --down_sample \
  --down_sample_rate 0.4 \
  --save_model
```

### Budget-aware Target-only Experiments
```angular2html
# Named Entity Recognition (NER) 
python code/ner_budget.py \
  --lm_model procbert \
  --data_name chemsyn \
  --gpu_ids 0,1  \
  --output_dir "$OUTPUT_DIR"/chemsyn \
  --learning_rate 1e-5 \
  --task_name ner \
  --batch_size 16 \
  --max_len 512 \
  --epochs 25 \
  --budget 700 \
  --save_model

# Relation Extraction (RE)
python code/rel_budget.py \
  --lm_model procbert \
  --data_name chemsyn \
  --gpu_ids 0,1  \
  --output_dir "$OUTPUT_DIR"/chemsyn \
  --learning_rate 1e-5 \
  --task_name rel \
  --batch_size 48 \
  --max_len 256 \
  --epochs 5 \
  --budget 700 \
  --down_sample \
  --down_sample_rate 0.4 \
  --save_model
```

### Auxiliary Experiments
```angular2html
# Named Entity Recognition (NER) 
python code/ner.py \
  --lm_model procbert \
  --data_name chemsyn \
  --gpu_ids 0,1  \
  --output_dir "$OUTPUT_DIR"/chemsyn \
  --learning_rate 1e-5 \
  --task_name ner \
  --batch_size 16 \
  --max_len 512 \
  --epochs 20 \
  --save_model

# Relation Extraction (RE)
python code/rel.py \
  --lm_model procbert \
  --data_name chemsyn \
  --gpu_ids 0,1  \
  --output_dir "$OUTPUT_DIR"/chemsyn \
  --learning_rate 1e-5 \
  --task_name rel \
  --batch_size 48 \
  --max_len 256 \
  --epochs 5 \
  --down_sample \
  --down_sample_rate 0.4 \
  --save_model
```

