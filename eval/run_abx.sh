PATH_TO_ABX_ITEMS="/data/home/xueyao/workspace/dataset/ABX_data"
DB_NAME="dev-clean"

# Kmeans centroids
for n_cluster in 32 64 128 256 512 1024
do
    PATH_FEATURE_DIR="/fsx-project/xueyao/data/kmeans_of_librilight_eval/c${n_cluster}"
    OUTPUT_DIR="/data/home/xueyao/workspace/libri-light/eval_results/c${n_cluster}"
    python eval_ABX.py $PATH_FEATURE_DIR  $PATH_TO_ABX_ITEMS/$DB_NAME.item \
        --file_extension ".pt" \
        --out $OUTPUT_DIR \
        --feature_size 0.02
done

# HuBERT-ASR PPG
PATH_FEATURE_DIR="/fsx-project/xueyao/data/ppg_of_librilight_eval"
OUTPUT_DIR="/data/home/xueyao/workspace/libri-light/eval_results/ppg"
python eval_ABX.py $PATH_FEATURE_DIR  $PATH_TO_ABX_ITEMS/$DB_NAME.item \
    --file_extension ".pt" \
    --out $OUTPUT_DIR \
    --feature_size 0.02