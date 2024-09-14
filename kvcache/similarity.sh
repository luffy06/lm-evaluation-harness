DEVICES=1
PROJECT_ROOT=$(git rev-parse --show-toplevel)
MODEL_PATH=/l/users/shangyu.wu/models/llama-3-8b
DATASET=wikitext-103

CUDA_VISIBLE_DEVICES=$DEVICES python $PROJECT_ROOT/kvcache/similarity.py \
    --dataset $DATASET \
    --split "test" \
    --model_name_or_path $MODEL_PATH \
    --batch_size 64 \
    --max_length 512 \
    --stride 512 \
    --ignore_self \
    --threshold 0.5
