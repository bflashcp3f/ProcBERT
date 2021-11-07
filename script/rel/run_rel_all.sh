
# Make sure OUTPUT_DIR is set properly.
# Parameter: model data_name batch_size epoch_num learning_rate max_sequence_len gpu_ids neg_sample_rate output_dir.

# ChemSyn
sh script/rel/run_rel.sh procbert chemsyn 48 5 1e-5 256 0,1 0.4 $OUTPUT_DIR

# PubMed
sh script/rel/run_rel.sh procbert pubmed 48 5 2e-5 256 0,1 1.0 $OUTPUT_DIR

# WLP
sh script/rel/run_rel.sh procbert wlp 128 5 2e-5 128 0,1 0.4 $OUTPUT_DIR