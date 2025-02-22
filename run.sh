for sub in qnli qqp rte sst2
do
    srun --gres gpu \
    --partition=csc413 \
    python3 test_by_name.py \
    --base_model "google/fnet-base" \
    --finetuned_model "gchhablani/fnet-base-finetuned-$sub" \
    --dataset "glue" \
    --subdata "$sub"
done