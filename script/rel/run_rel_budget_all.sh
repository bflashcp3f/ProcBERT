
OUTPUT_DIR='./output'

# Parameter: model budget data_name batch_size epoch_num learning_rate max_sequence_len gpu_ids down_sample_rate output_dir.
# ChemSyn
sh script/rel/run_rel_budget.sh procbert 700 chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_budget.sh procbert 1500 chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_budget.sh procbert 2300 chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# PubMed
sh script/rel/run_rel_budget.sh procbert 700 pubmed 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_budget.sh procbert 1500 pubmed 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# WLP
sh script/rel/run_rel_budget.sh procbert 700 wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_budget.sh procbert 1500 wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_budget.sh procbert 2300 wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR