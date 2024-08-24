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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-103")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name_or_path", type=str, default="llama-3-8b")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--last_area", type=float, default=0.2)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
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
    sparsity = [0 for _ in range(num_layers)]
    word_scores = {}
    word_freq = {}
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, data_len, stride)):
        end_loc = min(begin_loc + max_length, data_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(args.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            for token_id in input_ids[0]:
                token_id = token_id.item()
                if token_id not in word_freq:
                    word_freq[token_id] = 0
                    word_scores[token_id] = 0
                else:
                    word_freq[token_id] += 1
            outputs = model(input_ids, labels=target_ids, use_cache=True, output_hidden_states=True, output_attentions=True)
            attentions = outputs.attentions
            for layer_i, attention_i in enumerate(attentions):
                limits = torch.max(attention_i, dim=-1).values * args.threshold
                limits = limits.unsqueeze(-1).expand_as(attention_i)
                stat = attention_i <= limits
                ratio = stat.cpu().sum() / stat.numel()
                sparsity[layer_i] += ratio * 100
                scores = attention_i.squeeze().sum(0).sum(1)
                for token_id, score in zip(input_ids[0], scores):
                    word_scores[token_id.item()] += score.item()
                
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == data_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    logger.info(f"Perplexity: {ppl.item()}")
    sparsity = [s / len(nlls) for s in sparsity]

    word_ids = sorted(list(word_freq.keys()), key=lambda x: word_freq[x])
    sorted_word_scores = [word_scores[word_id] for word_id in word_ids]
    sorted_word_freq = [word_freq[word_id] for word_id in word_ids]
    last_idx = len(word_ids) - int(len(word_ids) * args.last_area)
    for word_id in word_ids[last_idx:]:
        logger.info(f"Word: {tokenizer.decode([word_id])}, Frequency: {word_freq[word_id]}, Score: {word_scores[word_id]}, Average Score: {word_scores[word_id] / word_freq[word_id]}")

    plt.figure()
    plt.plot(sparsity)
    plt.xlabel("Layer")
    plt.ylabel("Attention Sparsity")
    plt.ylim(0, 100)
    plt.savefig(f"attention-sparsity-{args.dataset}-{model_name}.png")

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = ax1.twinx()
    ax1.plot(sorted_word_freq, label="Frequency", c="grey")
    ax2.scatter([i for i in range(len(word_ids))], sorted_word_scores, label="Score", c="r", s=1)

    ax1.set_xlabel("Word Index")
    ax1.set_ylabel("Frequency", color="grey")
    ax2.set_ylabel("Attention Score", color="r")

    ax1 = fig.add_subplot(122)
    ax2 = ax1.twinx()
    ax1.plot(sorted_word_freq[last_idx:], label="Frequency", c="grey")
    ax2.scatter([i for i in range(len(sorted_word_scores[last_idx:]))], sorted_word_scores[last_idx:], label="Score", c="r", s=1)

    ax1.set_xlabel("Word Index")
    ax1.set_ylabel("Frequency", color="grey")
    ax2.set_ylabel("Attention Score", color="r")
    plt.savefig(f"attention-scores-{args.dataset}-{model_name}.png")
