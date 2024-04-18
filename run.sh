srun --gres gpu \
--partition=csc413 \
python3 test_by_name.py \
--base_model "distilbert/distilbert-base-uncased" \
--finetuned_model "Akash7897/distilbert-base-uncased-finetuned-sst2" \
--dataset "glue" \
--subdata "sst2"