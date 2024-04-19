for name in vicclab
do
    srun --gres gpu \
    --partition=csc413 \
    python3 test_by_name.py \
    --base_model "distilbert/distilbert-base-uncased" \
    --finetuned_model "$name/distilbert_sst2_finetuned" \
    --dataset "glue" \
    --subdata "sst2"
done