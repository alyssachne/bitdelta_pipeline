for sub in sst2 qnli qqp rte
do
    srun --gres gpu \
    --partition=csc413 \
    python3 test_by_name.py \
    --base_model "google-bert/bert-base-cased" \
    --finetuned_model "gchhablani/bert-base-cased-finetuned-$sub" \
    --dataset "glue" \
    --subdata "$sub"
done