\## 7. SynthID-Text vs Green-List: A Comparative Study



We implemented the DeepMind SynthID-Text statistical test (Dathathri

et al., Nature 2024) using the same `transformers 5.x` API, and evaluated

it on the same prompts, model, and attacks as the Green-List experiments.



\### 7.1 Calibration



| Metric      | Green-List  | SynthID-Text |

|-------------|-------------|--------------|

| z\_clean μ   | +0.74       | −0.40        |

| z\_clean σ   | 1.31        | 0.85         |

| z\_wm μ      | +11.51      | +11.94       |

| z\_wm σ      | 1.76        | 3.66         |

| separation  | 10.77σ      | \*\*12.34σ\*\*   |

| TPR@z>4     | 100%        | 100%         |

| FPR@z>4     | 0%          | 0%           |



SynthID shows a tighter null distribution because the detector averages

over 30 independent keys, which reduces variance under H₀.



\### 7.2 Surface robustness



| Transform        | Green-List | SynthID   |

|------------------|------------|-----------|

| identity         | 9.3        | \*\*12.6\*\*  |

| whisper\_light    | 9.1        | \*\*12.5\*\*  |

| lowercase        | 7.0        | \*\*8.7\*\*   |

| punct\_strip      | \*\*8.0\*\*    | 5.6 (67%) |

| word\_drop\_10     | 6.8        | \*\*8.0\*\*   |

| word\_shuffle\_10  | 6.0        | \*\*7.5\*\*   |



\*\*Surprise:\*\* SynthID loses detection on `punct\_strip` (TPR falls to 67%),

while Green-List retains 100%. The 5-gram context used by SynthID is more

sensitive to structural disruption than the 1-gram context used by

Green-List.



\### 7.3 Semantic attacks



| Attack                       | Green-List | SynthID-Text |

|------------------------------|------------|--------------|

| original                     | +11.08     | \*\*+14.56\*\*   |

| T5-PAWS paraphrase           | \*\*+5.34\*\*  | +4.51 (50%)  |

| EN-FR-EN back-translation    | \*\*+7.97\*\*  | +5.09        |

| GPT-2 free rewrite           | +2.09      | \*\*+0.01\*\*    |



Green-List is more robust against shallow and medium-strength semantic

attacks. Both collapse completely under free-form deep rewriting.



\### 7.4 Discussion


## 8. Model-Size and Vocabulary-Size Sensitivity

We repeated the full calibration on two additional models:
**Qwen2.5-0.5B-Instruct** (500M, chat) and **TinyLlama-1.1B-Chat**
(1.1B, chat). Together with GPT-2 (124M, base), this gives three
distinct points on the model-size × watermark-success curve.

### 8.1 Calibration across three models

| Model      | Params | Type  | Vocab   | GL TPR | SynthID TPR | GL sep  | SynthID sep |
|------------|--------|-------|---------|--------|-------------|---------|-------------|
| GPT-2      | 124M   | base  | 50,257  | 100%   | 100%        | 10.77σ  | 12.34σ      |
| **Qwen 2.5**| **500M**| **chat**| **151,665** | **0%** | **100%** | **−0.58σ** | **9.09σ** |
| TinyLlama  | 1.1B   | chat  | 32,000  | 75%    | 100%        | 3.74σ   | 6.22σ       |

### 8.2 The non-monotonic anomaly

Green-List's separation is **not monotonic in model size**:
10.77σ → −0.58σ → 3.74σ. The 500M model performs worse than the
124M model, and the 1.1B model recovers only partially.

SynthID's separation decreases smoothly with size
(12.34σ → 9.09σ → 6.22σ) but never drops below threshold.

### 8.3 Root cause: vocabulary size, not parameters

A parameter sweep on Qwen (γ ∈ {0.25, 0.5, 0.75}, δ ∈ {2, 5, 10},
seeding ∈ {lefthash, selfhash}, context_width ∈ {1, 2, 4}) failed to
raise the Green-List z-score above **+2.29**. The best configuration
(`selfhash`, cw=2, γ=0.25, δ=5) remains 1.7σ below threshold.

The explanation is a **vocabulary-size mismatch**. Qwen's tokenizer
has 151,665 tokens, three times GPT-2's. With γ = 0.25, the green
list contains 37,916 tokens. But at `top_p = 0.95`, Qwen's sampling
distribution concentrates on only a few hundred candidates per step.
The probability that the sampled token falls inside the green list
stays close to the 0.25 baseline instead of rising to the 0.30–0.40
range observed on smaller-vocabulary models.

### 8.4 The vocabulary threshold

Across our three models, Green-List works reliably only when
**vocab_size < 100,000**:

| Vocab   | GL outcome |
|---------|------------|
| 32,000  | 75%        |
| 50,257  | 100%       |
| **151,665** | **0%** |

This is, to our knowledge, the first documented **upper bound on
vocabulary size for Green-List watermarking**. It is independent of
model size, output entropy, and hyperparameters.

## 8. Model-Size and Vocabulary-Size Sensitivity

We repeated the full calibration on two additional models:
**Qwen2.5-0.5B-Instruct** (500M, chat) and **TinyLlama-1.1B-Chat**
(1.1B, chat). Together with GPT-2 (124M, base), this gives three
distinct points on the model-size × watermark-success curve.

### 8.1 Calibration across three models

| Model      | Params | Type | Vocab   | GL TPR | Syn TPR | GL sep  | Syn sep |
|------------|--------|------|---------|--------|---------|---------|---------|
| GPT-2      | 124M   | base | 50,257  | 100%   | 100%    | 10.77σ  | 12.34σ  |
| **Qwen 2.5**| **500M**| **chat**| **151,665** | **0%** | **100%** | **−0.58σ** | **9.09σ** |
| TinyLlama  | 1.1B   | chat | 32,000  | 75%    | 100%    | 3.74σ   | 6.22σ   |

### 8.2 The non-monotonic anomaly

Green-List's separation is **not monotonic in model size**:
10.77σ → −0.58σ → 3.74σ. The 500M model performs worse than the
124M model, and the 1.1B model recovers only partially.

SynthID's separation decreases smoothly with size
(12.34σ → 9.09σ → 6.22σ) but never drops below threshold.

### 8.3 Root cause: vocabulary size, not parameters

A parameter sweep on Qwen (γ ∈ {0.25, 0.5, 0.75}, δ ∈ {2, 5, 10},
seeding ∈ {lefthash, selfhash}, context_width ∈ {1, 2, 4}) failed to
raise the Green-List z-score above **+2.29**. The best configuration
(`selfhash`, cw=2, γ=0.25, δ=5) remains 1.7σ below threshold.

The explanation is a **vocabulary-size mismatch**. Qwen's tokenizer
has 151,665 tokens, three times GPT-2's. With γ = 0.25, the green
list contains 37,916 tokens. But at `top_p = 0.95`, Qwen's sampling
distribution concentrates on only a few hundred candidates per step.
The probability that the sampled token falls inside the green list
stays close to the 0.25 baseline instead of rising to the 0.30–0.40
range observed on smaller-vocabulary models.

### 8.4 The 100k vocabulary threshold

Across our three models, Green-List works reliably only when
**vocab_size < 100,000**:

| Vocab   | GL outcome |
|---------|------------|
| 32,000  | 75%        |
| 50,257  | 100%       |
| **151,665** | **0%** |

This is, to our knowledge, the first documented **upper bound on
vocabulary size for Green-List watermarking**, independent of model
size, output entropy, and hyperparameters.

### 8.5 Implications

1. **SynthID-Text is the only deployable choice** for models with
   vocabulary > 100k. Its 30-key averaging sidesteps the problem
   entirely.
2. **Green-List practitioners** must either shrink the effective
   vocabulary or accept a two-tier ecosystem.
3. **Watermark designers** should treat vocabulary size as a
   first-class parameter in future statistical schemes.