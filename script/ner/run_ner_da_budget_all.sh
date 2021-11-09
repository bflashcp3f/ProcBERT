
OUTPUT_DIR='./output'

# Parameter: model budget src_data tgt_data batch_size epoch_num learning_rate max_sequence_len gpu_ids output_dir.
# PubMed -> ChemSyn
sh script/ner/run_ner_da_budget.sh procbert 700 pubmed chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 1500 pubmed chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 2300 pubmed chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR

# WLP -> ChemSyn
sh script/ner/run_ner_da_budget.sh procbert 700 wlp chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 1500 wlp chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 2300 wlp chemsyn 16 25 1e-5 512 0,1 $OUTPUT_DIR

# WLP -> PubMed
sh script/ner/run_ner_da_budget.sh procbert 700 wlp pubmed 16 35 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 1500 wlp pubmed 16 35 1e-5 512 0,1 $OUTPUT_DIR

# ChemSyn -> PubMed
sh script/ner/run_ner_da_budget.sh procbert 700 chemsyn pubmed 16 35 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 1500 chemsyn pubmed 16 35 1e-5 512 0,1 $OUTPUT_DIR

# ChemSyn -> WLP
sh script/ner/run_ner_da_budget.sh procbert 700 chemsyn wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 1500 chemsyn wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 2300 chemsyn wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR

# PubMed -> WLP
sh script/ner/run_ner_da_budget.sh procbert 700 pubmed wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 1500 pubmed wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR
sh script/ner/run_ner_da_budget.sh procbert 2300 pubmed wlp 16 25 1e-5 512 0,1 $OUTPUT_DIR