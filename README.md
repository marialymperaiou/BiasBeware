# BiasBeware

A dataset containing product descriptions which have been rephrased using cognitive biases.

![BiasBeware painting showing preference for the DSLR with a 25% discount over a non-discounted DSLR](teaser.png)

**When words nudge decisions**

Modern recommendation systems don’t just process products—they process persuasion. BiasBeware investigates how subtle linguistic cues influence LLM-driven recommendations, and challenges models to detect, explain, and remove these effects.


## Subtask C — Sanitization of Attacked Product Descriptions

Can a system **remove manipulation without removing meaning**?

In **Subtask C**, participants are given product descriptions that have been subtly attacked with cognitive-bias cues. The goal is to **rewrite** these descriptions so that the manipulative signal is removed, while the **factual product content stays intact**.

This is a **constrained debiasing** task: a successful system should produce text that is:
- **less manipulative** than the attacked version,
- **faithful** to the original product,
- and **minimally edited**, rather than fully rewritten into a new description.

In other words, the ideal system should keep *what the product is*, while removing *how it is being nudged*.

### What we evaluate

We evaluate sanitization along two complementary dimensions:

#### 1. Downstream debiasing success
Does the rewritten description reduce the ranking distortion caused by the attack?

We measure this using average rank displacement:

```
avg|Δ|
```

Lower is better.  
A value of **0** means the attacked product’s position has been fully restored to the neutral baseline.

#### 2. Distributional faithfulness to the clean reference
Does the rewritten text move back toward the original neutral description distribution?

We measure this using **KL divergence** between the token distribution of the sanitized set and the original clean set:

```
KL(initial || sanitized)
```

where:
- \(P_{\text{initial}}\) = the distribution of the original clean descriptions
- \(P_{\text{sanitized}}\) = the distribution of the participant’s rewritten descriptions

**Interpretation:**  
- \(D_{\mathrm{KL}} = 0\) means the two distributions are identical
- Lower values mean the sanitized descriptions are lexically closer to the clean reference
- Higher values mean more drift from the original neutral style

Complementarily: We also report **Jensen–Shannon divergence (JSD)** as a symmetric, bounded companion metric:

```
JSD(initial || sanitized)

0 ≤ JSD(initial, sanitized) ≤ 1
```

- **0** = identical distributions  
- values closer to **0** = highly similar  
- values closer to **1** = highly different

Both KL and JSD calculation functions are provided.

### Pilot Baseline

As an initial baseline, we debias our sample data using **Gemini 3 (zero-shot)**.

> **Prompt**:
```
Debias the following product descriptions
```

Results:

- **KL(initial || Gemini debias) = 0.197**
- **JSD(initial, Gemini debias) = 0.059**

### How to read these numbers

These values are encouragingly low:

- **KL = 0.197** means the debiased descriptions are still fairly close to the original clean distribution, though not identical.
- **JSD = 0.059** means the overall lexical distributions are very similar. Since JSD ranges from **0** to **1**, a value of **0.059** indicates only a small amount of distributional drift.

In simple terms:  
**the baseline is able to remove some bias while staying reasonably close to the original clean style**.

### Subtask takeaways

Subtask C is about more than rewriting text — it is about **restoring neutrality without destroying content**.

Participants must learn how to:
- identify and remove subtle manipulative cues,
- preserve product facts,
- and bring model behavior back toward the neutral baseline.

This makes Subtask C a benchmark for **controlled debiasing, faithful rewriting, and robust recommendation repair**.

# Paper reference

For reference, check our EMNLP 2025 paper:

**Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations**  

Giorgos Filandrianos, Angeliki Dimitriou, Maria Lymperaiou, Konstantinos Thomas, Giorgos Stamou  

- [ACL Anthology page](https://aclanthology.org/2025.emnlp-main.1140/)

If you find this work useful, please cite:

```bibtex
@inproceedings{filandrianos-etal-2025-bias,
  title = {Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations},
  author = {Filandrianos, Giorgos and Dimitriou, Angeliki and Lymperaiou, Maria and Thomas, Konstantinos and Stamou, Giorgos},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year = {2025},
  pages = {22397--22426},
  url = {https://aclanthology.org/2025.emnlp-main.1140/}
}
```
