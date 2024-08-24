DEVICES=5
PROJECT_ROOT=$(git rev-parse --show-toplevel)
MODEL_PATH=/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Shangyu.Wu@mbzuai.ac.ae/models/opt-6.7b
DATASET=wikitext-103

CUDA_VISIBLE_DEVICES=$DEVICES python $PROJECT_ROOT/kvcache/attention_sparsity.py \
    --dataset $DATASET \
    --split "test" \
    --model_name_or_path $MODEL_PATH \
    --batch_size 64 \
    --max_length 512 \
    --stride 512 \
    --threshold 0.01 \
    --last_area 0.2