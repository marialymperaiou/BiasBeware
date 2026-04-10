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

Nevertheless, in the consequent pilot study, we prove that even though recognizing cognitive biases is a rather simple job for humans (at least when they have been casually exposed to them), it poses a significant burden for LLM recommenders. Our executed baselines showcase that:

A. It is significantly hard even for potent models (Gemini 3 zero-shot) to distinguish the **causes behind a recommendation outcome**, and thus indicate the product responsible for interfering with a fair (unbiased) ranking behavior (Subtask A).

B. As proven also in our prior EMNLP paper, defending against attacks stemming from cognitive biases is not a trivial endeavor. In our study, we leverage a small LLM recommender, Qwen3-0.6B, which despite its small size, it is a capable recommender in the unbiased setup. By attempting to defend in a simple, prompt-driven way, we realize that **generic intruction-driven prompting is not sufficient** for blocking biased influence, as evidenced by large rank changes after attack (and despite defensible prompt usage) in comparison to the initial, bias-free recommendation setup (Subtask B).

C. Debiasing biased product descriptions remains a persistent concern, since it requires balancing between preserving product identity and rewriting biased parts without references. When leveraging a subset of cognitive biases in their most obvious form (higher intensity - more prominent bias), Gemini 3 is capable of **partially restoring fair descriptions** to an acceptable extend, although there is still room for improvement. However, is this still going to be valid beyond our simple sample data? (Subtask C)

---

## Subtask A — Causal Attribution of Recommendation Outcomes

Can a system **identify which biased product is causally connected to a recommendation outcome**?

In **Subtask A**, participants are given **pairs of attacked product descriptions**. Each pair contains two products, denoted:

- **Description A**
- **Description B**

Each description comes from a product that has been manipulated using a cognitive bias. The task is to determine whether the observed recommendation outcome is better attributed to:

- **A**
- **B**
- **Uncertain**

This is a **causal attribution** task: the goal is not to decide which product is better in general, but rather to infer **which manipulated description is more likely to explain the downstream recommendation effect**.

### How the pilot data is constructed

For the pilot study, we begin from a clean control ranking and several attack-specific recommendation settings. The clean setting is derived from the neutral binoculars file, while attack settings are derived from the corresponding bias-injected files. The control and attacked files contain product lists together with model ranking outputs and alignment mappings between output rank and source product position. 

We use **ChatGPT 5.4** as the recommender. For each attacked recommendation setting, we compare the attacked product’s position against its position in the control ranking and assign one of three movement labels:

- **Up**: the attacked product moved higher in the ranking than in control
- **Down**: the attacked product moved lower in the ranking than in control
- **Same**: the attacked product stayed in the same ranking position as in control

We then construct pairs of attacked products, usually from different cognitive-bias families, and keep for each:

- the attacked product description
- its cognitive bias type
- its movement label (`Up`, `Down`, `Same`)

From these paired attacked products, we create the final pilot instances shown to participants:

- `description_A`
- `description_B`
- `final_label`

The `final_label` is generated as follows:

- if the two movement labels differ, one side is randomly selected as the gold causal source
- if the two movement labels are the same, the gold label is **Uncertain**

This produces a controlled causal-attribution benchmark where the participant must recover which product better explains the observed recommendation effect.

### Pilot setup

For the pilot, we randomly sampled **100 paired examples**.

Model roles:
- **ChatGPT 5.4**: recommender
- **Gemini 3**: causal-attribution model

Gemini 3 receives a pair of attacked product descriptions and must output exactly one of:

- `A`
- `B`
- `Uncertain`

### What we evaluate

We evaluate predictions against the gold labels using standard multi-class classification metrics.

Primary metric:
- **Macro-F1**

Additional metrics:
- **Accuracy**
- **Balanced Accuracy**
- **Per-class Precision / Recall / F1**
- **Confusion Matrix**

We also report one-vs-rest ROC/AUC values for completeness, although these are less informative when computed from hard class predictions rather than calibrated probabilities.

### Pilot results

On the 100-example pilot set, **Gemini 3** achieved:

- **Macro-F1:** **0.153**
- **Accuracy:** **0.160**
- **Balanced Accuracy:** **0.203**

Per-class performance:

- **A**
  - Precision: **0.167**
  - Recall: **0.316**
  - F1: **0.218**

- **B**
  - Precision: **0.167**
  - Recall: **0.273**
  - F1: **0.207**

- **Uncertain**
  - Precision: **0.100**
  - Recall: **0.021**
  - F1: **0.034**

Confusion matrix:

|                | pred_A | pred_B | pred_Uncertain |
|----------------|-------:|-------:|---------------:|
| **true_A**         | 6 | 13 | 0 |
| **true_B**         | 15 | 9 | 9 |
| **true_Uncertain** | 15 | 32 | 1 |

One-vs-rest ROC AUC:

- **A:** **0.473**
- **B:** **0.301**
- **Uncertain:** **0.424**
- **Macro OVR AUC:** **0.399**
- **Weighted OVR AUC:** **0.392**

### How to read these numbers

These pilot results show that **causal attribution is substantially harder than bias identification**.

Unlike the earlier annotation stage, where humans reached perfect agreement on explicit bias cues, Subtask A requires the model to reason about a **downstream recommendation effect** and decide which attacked product is most likely responsible for it. This is a much harder causal inference problem.

A few observations stand out:

- Gemini 3 performs **well below a strong level of reliability** on this task.
- It performs somewhat better on **A** and **B** than on **Uncertain**.
- The **Uncertain** class is especially difficult, with very low recall, suggesting that the model tends to over-commit to one side rather than abstain when attribution is ambiguous.
- Performance differences across classes indicate that causal-attribution behavior should be analyzed classwise, not only through overall accuracy.

### Pilot takeaway

The pilot confirms that **Subtask A is non-trivial and meaningful**: even a strong language model struggles to identify which biased product is causally connected to the recommendation outcome.

This is encouraging for the benchmark design. It suggests that:
- the task is not solved by superficial pattern matching,
- causal attribution is meaningfully distinct from recommendation,
- and future systems should be evaluated not only on whether they detect bias, but on whether they can correctly **attribute downstream recommendation effects** to the responsible manipulated input.

Overall, the pilot supports Subtask A as a challenging benchmark for **causal reasoning under biased language conditions**.

## Subtask B — Defense Against Attack

Can a system **maintain fair recommendations under manipulated descriptions**?

In **Subtask B**, participants are given recommendation settings with competing products where one or more descriptions may have been attacked using cognitive-bias cues. The model participants are required to run as recommender is **Qwen3-0.6B**, which can be executed in commonly available environments such as Google Colab or Kaggle GPUs. The recommendation code will also be provided. In addition, the original pre-attack ranking is given, and the objective is to reduce unfair rank shifts caused by manipulated language.

This is a **robustness** task: a successful system should:
- detect or neutralize the effect of biased product descriptions,
- preserve the integrity of the original recommendation ordering,
- remain robust across different recommendation settings and attack styles,
- and generalize beyond a single prompt-specific defense trick.

### What we evaluate

We evaluate defense through rank restoration using **sum of squared rank displacement**:

where

\[
\sum_{i=1}^{n}\left(r_{\mathrm{before}}(i)-r_{\mathrm{after}}(i)\right)^2
\]

Lower is better.  
Values closer to **0** indicate that the defended system keeps the post-attack ranking closer to the original unbiased ordering, while also penalizing larger ranking deviations more strongly.

To provide additional descriptive analysis, we also report:
- **avg|Δ|**: average absolute rank displacement,
- **Spearman correlation** between the defended and original rankings,
- **Kendall tau** correlation,
- and **Kendall distance**.

These complementary metrics help distinguish between small local shifts and more global ranking disruption.

### Pilot observations

We conducted preliminary **Subtask B** experiments using **Qwen3-0.6B** together with a simple prompt-based defense. Results show that this lightweight defense is **not sufficient to reliably preserve the original ranking** under cognitive-bias attacks.

Across all attack families, the overall pilot averages are:

- **avg|Δ| = 2.5755**
- **sum(Δ²) = 68.2583**
- **Spearman correlation = 0.0081**
- **Kendall tau = 0.0081**
- **Kendall distance = 0.4959**

These values suggest that, even after defense, rankings remain substantially distorted. In particular:
- the average product still moves by about **2.6 ranking positions**,
- pairwise order agreement remains very weak,
- and the defended rankings stay far from the original unbiased ordering.

### Differences across attack families

At the attack-family level, the defense performs similarly poorly overall, but some patterns emerge:

- **Scarcity** is the hardest family by the primary metric:
  - **sum(Δ²) = 69.5125**
- **Discount framing** is slightly less disruptive on average:
  - **sum(Δ²) = 67.5062**
- **Exclusivity** is very close in aggregate:
  - **sum(Δ²) = 67.6424**
- **Social proof** is the comparatively easiest family for the current defense:
  - **sum(Δ²) = 68.3375**
  - **Spearman = 0.0514**
  - **Kendall tau = 0.0404**
  - **Kendall distance = 0.4798**

Even so, none of these values indicate strong recovery of the original ranking.

### Variation across individual attacks

Within each family, some attack instances are much more disruptive than others.

Examples:
- **Discount framing / Attack 0**:
  - **sum(Δ²) = 86.00**
  - **Spearman = -0.1644**
  - **Kendall tau = -0.1111**
- **Scarcity / Attack 6**:
  - **sum(Δ²) = 78.35**
- **Exclusivity / Attack 2**:
  - **sum(Δ²) = 50.65**
  - one of the mildest cases in the pilot
- **Social proof / Attack 0**:
  - **sum(Δ²) = 56.45**
  - **Spearman = 0.3290**
  - **Kendall tau = 0.2603**
  - the most recoverable case under the current defense

This variability suggests that some biased interventions are easier to counteract than others, even within the same family.

### Pilot takeaway

The pilot confirms that **Subtask B is non-trivial and meaningful**. A simple defense prompt does not reliably restore fair recommendation behavior: products still move substantially, and overall ranking agreement remains weak.

This is encouraging for the benchmark design. It shows that:
- the task is not solved by generic instruction-based prompting,
- robust defense requires more than superficial prompt hardening,
- and future systems should be evaluated on whether they can truly preserve the original ranking structure under biased language attacks.

Overall, the pilot supports Subtask B as a challenging benchmark for **robust recommendation under cognitively manipulated language**.
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

# Check our website

[BiasBeware website](https://marialymperaiou.github.io/BiasBeware/)

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
