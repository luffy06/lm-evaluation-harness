import torch
import logging
import argparse
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM
)
from datasets import load_dataset
from tqdm import tqdm
from transformers.cache_utils import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityCache(Cache):
    pass

def compute_similarity(a, b, threshold=0.5):
    score = torch.matmul(a, b.transpose(-1, -2))
    score = torch.softmax(score, dim=-1)
    sim = score > threshold
    return sim.sum()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-103")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3-8b")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ignore_self", action="store_true")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    num_layers = config.num_hidden_layers
    model_name = config.model_type
    logger.info(config)

    test = load_dataset("wikitext", args.dataset + "-raw-v1", split=args.split)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = args.max_length
    stride = args.stride
    data_len = encodings.input_ids.size(1)

    nlls = []
    sim_keys = torch.zeros(num_layers, num_layers)
    sim_values = torch.zeros(num_layers, num_layers)
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, data_len, stride)):
        end_loc = min(begin_loc + max_length, data_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(args.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, use_cache=True, output_hidden_states=True, output_attentions=True)
            kv_caches = outputs.past_key_values
            for layer_i, kv_cache_i in enumerate(kv_caches):
                for layer_j, kv_cache_j in enumerate(kv_caches):
                    batch_size, num_heads, seq_len, head_dim = kv_cache_i[0].shape
                    sim_key = compute_similarity(kv_cache_i[0], kv_cache_j[0], threshold=args.threshold)
                    sim_value = compute_similarity(kv_cache_i[1], kv_cache_j[1], threshold=args.threshold)
                    sim_key = sim_key.cpu() / (num_heads * seq_len * seq_len)
                    sim_value = sim_value.cpu() / (num_heads * seq_len * seq_len)
                    sim_keys[layer_i, layer_j] += sim_key
                    sim_values[layer_i, layer_j] += sim_value

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == data_len:
            break

    sim_keys = sim_keys / len(nlls)
    sim_values = sim_values / len(nlls)

    if args.ignore_self:
        for i in range(num_layers):
            sim_keys[i, i] = 0
            sim_values[i, i] = 0

    ppl = torch.exp(torch.stack(nlls).mean())
    logger.info(f"Perplexity: {ppl.item()}")

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(sim_keys.cpu().numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Key Similarity of {model_name} on {args.dataset}")
    plt.xlabel("Layer")
    plt.ylabel("Layer")

    plt.subplot(1, 2, 2)
    plt.imshow(sim_values.cpu().numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Value Similarity of {model_name} on {args.dataset}")
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.show()

    plt.savefig(f"kv-sim-{args.dataset}-{model_name}-{int(args.threshold*100)}.png")