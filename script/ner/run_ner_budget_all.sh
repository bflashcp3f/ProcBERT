
OUTPUT_DIR='./output'

# Parameter: model budget data_name batch_size epoch_num learning_rate max_sequence_len gpu_ids output_dir.
# ChemSyn
sh script/ner/run_ner_budget.sh procbert 700 chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_budget.sh procbert 1500 chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_budget.sh procbert 2300 chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR

# PubMed
sh script/ner/run_ner_budget.sh procbert 700 pubmed 16 50 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_budget.sh procbert 1500 pubmed 16 50 1e-5 512 0,1 $OUTPUT_DIR

# WLP
sh script/ner/run_ner_budget.sh procbert 700 wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_budget.sh procbert 1500 wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_budget.sh procbert 2300 wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR