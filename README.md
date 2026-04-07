# BiasBeware

A dataset containing product descriptions which have been rephrased using cognitive biases.

![BiasBeware painting showing preference for the DSLR with a 25% discount over a non-discounted DSLR](teaser.png)

**When words nudge decisions**

Modern recommendation systems don’t just process products—they process persuasion. BiasBeware investigates how subtle linguistic cues influence LLM-driven recommendations, and challenges models to detect, explain, and remove these effects.


## Data Annotation Process

To construct a controlled benchmark for cognitive bias in product descriptions, we adopt a **two-stage annotation pipeline** combining automatic generation and human validation.

### 1. Neutralization of Product Descriptions

We start from raw Amazon product descriptions, which often contain implicit promotional language (e.g., *“best-selling”*, *“limited offer”*, *“highly rated”*).  
To enable controlled experimentation, we transform these into **neutral, feature-centered descriptions**.

This step involves:
- automatic rewriting using LLMs, and  
- manual verification to ensure:
  - removal of persuasive cues  
  - preservation of factual product attributes  

The resulting descriptions serve as the **unbiased reference set**.

---

### 2. Bias Injection

From each neutral description, we generate one or more **bias-injected variants** by introducing minimal linguistic cues corresponding to specific cognitive biases.

We consider the following bias taxonomy:

- **Social Proof** (e.g., “popular choice”, “trusted by many”)  
- **Scarcity** (e.g., “limited stock available”)  
- **Exclusivity** (e.g., “limited edition”, “exclusive version”)  
- **Authority** (e.g., “recommended by experts”)  
- **Contrast Effect** (e.g., comparisons highlighting relative advantage)  
- **Storytelling** (e.g., narrative framing around product use)  
- **Denomination Neglect** (e.g., framing prices to reduce perceived cost magnitude)  
- **Identity Signaling** (e.g., appealing to user identity or status)  
- **Decoy Effect** (e.g., introducing a less attractive alternative to influence choice)  
- **Discount Framing** (e.g., “save 20%”, “special deal”)  

Each manipulation is designed to be **minimal and localized**, ensuring that the underlying product content remains unchanged while introducing a targeted persuasive signal.

---

### 3. Human Annotation

Each description is annotated by **four independent annotators**.  
Annotators are asked to identify the **dominant bias category** present in the text from the predefined taxonomy, or assign:

- `no_bias` if no clear persuasive signal is present

Annotations are performed at the **description level**, with one label per item.

---

### 4. Agreement and Quality Control

To ensure annotation quality, we measure inter-annotator agreement using:

- **Fleiss’ κ** (multi-annotator agreement)  
- **Krippendorff’s α** (robust agreement metric for nominal labels)

Disagreements are resolved through:
- majority voting, and  
- manual adjudication for ambiguous or borderline cases.

This process ensures that bias annotations are **consistent, interpretable, and reliable**.

---

### 5. Final Dataset Structure

Each instance in the dataset contains:

- a **neutral (unbiased) description**  
- one or more **bias-injected variants**  
- a **bias label** (from the taxonomy above)  
- optional metadata (e.g., product category)

This structured setup enables controlled evaluation across all subtasks:
- attribution (Subtask A),
- defense (Subtask B),
- and sanitization (Subtask C).

---

### Summary

Our annotation pipeline ensures that:
- bias signals are **explicit yet minimally invasive**,  
- product descriptions remain **factually grounded**, and  
- annotations are **high-quality and reproducible**.

This enables precise analysis of how cognitive biases propagate through language into LLM-driven recommendation systems.

---
### Pilot data annotation

We embed cognitive biases in the most explicit way, to evaluate the validity of downstream baselines on the most 'obviously'  biased descriptions. Our 4 trained annotators achieve perfect agreement (100\% Fleiss’ κ and Krippendorff’s α), showing that the cognitive bias identification stage is easy for humans exposed in such information. This stage confirms that ChatGPT 5.4 did an excellent job in imbuing cognitive biases within product descriptions.

---

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
