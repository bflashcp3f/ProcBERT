
OUTPUT_DIR='./output'

# Parameter: model data_name batch_size epoch_num learning_rate max_sequence_len gpu_ids output_dir.
# ChemSyn
sh script/ner/run_ner.sh procbert chemsyn 16 20 1e-5 512 0,1 $OUTPUT_DIR

# PubMed
sh script/ner/run_ner.sh procbert pubmed 16 60 1e-5 512 0,1 $OUTPUT_DIR

# WLP
sh script/ner/run_ner.sh procbert wlp 32 20 1e-5 256 0,1 $OUTPUT_DIR