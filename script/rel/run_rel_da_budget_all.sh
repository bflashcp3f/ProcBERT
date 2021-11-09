
OUTPUT_DIR='./output'

# Parameter: model budget src_data tgt_data batch_size epoch_num learning_rate max_sequence_len gpu_ids down_sample_rate output_dir.
# PubMed -> ChemSyn
sh script/rel/run_rel_da_budget.sh procbert 700 pubmed chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 1500 pubmed chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 2300 pubmed chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# WLP -> ChemSyn
sh script/rel/run_rel_da_budget.sh procbert 700 wlp chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 1500 wlp chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 2300 wlp chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# WLP -> PubMed
sh script/rel/run_rel_da_budget.sh procbert 700 wlp pubmed 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 1500 wlp pubmed 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# ChemSyn -> PubMed
sh script/rel/run_rel_da_budget.sh procbert 700 chemsyn pubmed 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 1500 chemsyn pubmed 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# ChemSyn -> WLP
sh script/rel/run_rel_da_budget.sh procbert 700 chemsyn wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 1500 chemsyn wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 2300 chemsyn wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# PubMed -> WLP
sh script/rel/run_rel_da_budget.sh procbert 700 pubmed wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 1500 pubmed wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR
sh script/rel/run_rel_da_budget.sh procbert 2300 pubmed wlp 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR