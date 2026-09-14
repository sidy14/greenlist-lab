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



Neither watermark dominates the other. Their strengths are complementary:



\- \*\*SynthID\*\* achieves higher z-scores on the original text and is more

&#x20; robust to local token perturbations.

\- \*\*Green-List\*\* is more robust to structural changes (punctuation

&#x20; removal, back-translation, shallow paraphrase) because it uses a

&#x20; single-token context.



\*\*Design implication:\*\* a hybrid detector that combines the two tests

(or a defender that chooses based on the expected attack surface) would

outperform either alone.

