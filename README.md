| Broken by deep rewrite       | \*\*yes\*\*         |



\## SynthID-Text comparison (same GPT-2, same prompts)



| Metric              | Green-List | SynthID-Text |

|---------------------|------------|--------------|

| Calibration separation | 10.77σ  | \*\*12.34σ\*\*   |

| punct\_strip TPR     | 100%       | 67%          |

| T5-PAWS paraphrase   | +5.34      | +4.51        |

| Back-translation    | +7.97      | +5.09        |

| Free rewrite        | +2.09      | +0.01        |



\*\*Finding:\*\* neither watermark dominates. SynthID is stronger on

calibration and local perturbations; Green-List is more robust to

structural changes. Both fail against free-form rewriting.

