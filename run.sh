for sub in qnli qqp rte sst2 stsb
do
    srun --gres gpu \
    --partition=csc413 \
    python3 test_by_name.py \
    --base_model "google/fnet-large" \
    --finetuned_model "gchhablani/fnet-large-finetuned-$sub" \
    --dataset "glue" \
    --subdata "$sub"
done