import re
import math
import json
from collections import Counter

def load_descriptions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokenize(text):
    text = text.lower()
    # simple word tokenizer
    return re.findall(r"\b\w+\b", text)

def token_counts(descriptions):
    counts = Counter()
    for desc in descriptions:
        counts.update(tokenize(desc))
    return counts

def kl_divergence_from_counts(counts_p, counts_q, alpha=1.0):
    """
    Compute KL(P || Q) with Laplace smoothing.
    alpha=1.0 is standard add-one smoothing.
    """
    vocab = set(counts_p) | set(counts_q)

    total_p = sum(counts_p.values()) + alpha * len(vocab)
    total_q = sum(counts_q.values()) + alpha * len(vocab)

    kl = 0.0
    for term in vocab:
        p = (counts_p.get(term, 0) + alpha) / total_p
        q = (counts_q.get(term, 0) + alpha) / total_q
        kl += p * math.log(p / q)
    return kl

# Example usage
    
descriptions_a = load_descriptions("unbiased_set.json")
descriptions_b = load_descriptions("debiased_set.json")

counts_a = token_counts(descriptions_a)
counts_b = token_counts(descriptions_b)

kl_a_b = kl_divergence_from_counts(counts_a, counts_b)
kl_b_a = kl_divergence_from_counts(counts_b, counts_a)

print("KL(unbiased || debiased):", kl_a_b) # this is primarily used for evaluation
print("KL(debiased || unbiased):", kl_b_a)

# how much of the initial (unbiased) target distribution is still not recovered after debias?
# this is measured by KL(debiased || unbiased), therefore this is the primary evaluation metric
