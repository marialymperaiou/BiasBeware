from kl_eval import load_descriptions, token_counts, kl_divergence_from_counts
import math

def jensen_shannon_divergence_from_counts(counts_p, counts_q, alpha=1.0, base=2):
    vocab = set(counts_p) | set(counts_q)

    total_p = sum(counts_p.values()) + alpha * len(vocab)
    total_q = sum(counts_q.values()) + alpha * len(vocab)

    jsd = 0.0
    for term in vocab:
        p = (counts_p.get(term, 0) + alpha) / total_p
        q = (counts_q.get(term, 0) + alpha) / total_q
        m = 0.5 * (p + q)

        jsd += 0.5 * p * math.log(p / m) + 0.5 * q * math.log(q / m)

    if base != math.e:
        jsd /= math.log(base)

    return jsd


descriptions_a = load_descriptions("unbiased_set.json")
descriptions_b = load_descriptions("chatgpt_debias.json")

counts_a = token_counts(descriptions_a)
counts_b = token_counts(descriptions_b)

jsd = jensen_shannon_divergence_from_counts(counts_a, counts_b)

print("JSD(A, B):", jsd)