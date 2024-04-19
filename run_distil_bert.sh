for sub in sst2
do
    srun --gres gpu \
    --partition=csc413 \
    python3 test_by_name.py \
    --base_model "distilbert/distilbert-base-uncased" \
    --finetuned_model "avneet/distilbert-base-uncased-finetuned-$sub" \
    --dataset "glue" \
    --subdata "$sub"
done