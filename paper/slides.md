# When Watermarks Fail — 12-slide deck

## Slide 1 — Title
**When Watermarks Fail: Model-Size Sensitivity in Statistical LLM Watermarking**
Sidy Cissokho — 2026

## Slide 2 — Problem
- LLMs generate text indistinguishable from human writing.
- Statistical watermarks promise automatic provenance.
- Two schemes dominate: Green-List (Kirchenbauer 2023), SynthID-Text (DeepMind 2024).
- **No public head-to-head comparison exists.**

## Slide 3 — Green-List in one slide
- Partition vocabulary: green (γ=0.25) vs red, seeded by hash of previous token + secret key.
- Bias green logits by δ=2.0 at sampling.
- Detection: count green tokens, compute z-score.
- Under H₀: z ~ N(0,1). Threshold: z > 4.

## Slide 4 — SynthID-Text in one slide
- Continuous score g ∈ [0,1] per token, sampled from table indexed by n-gram.
- Average over K=30 independent secret keys.
- Null mean g = 0.5. Detection: (mean_g − 0.5) / SE.
- **Multi-key averaging reduces variance dramatically.**

## Slide 5 — Experimental setup
- Models: GPT-2 (124M), TinyLlama-1.1B-Chat (1.1B)
- Prompts: 8 (science, civics, education)
- Attacks: 6 surface + 4 semantic
- Threshold: z > 4 (FPR < 3e-5 per document)

## Slide 6 — Calibration: the headline table
| Model | Watermark | Separation | TPR |
|---|---|---|---|
| GPT-2 | Green-List | 10.77σ | 100% |
| GPT-2 | SynthID | 12.34σ | 100% |
| TinyLlama | Green-List | **3.74σ** | **75%** |
| TinyLlama | SynthID | 6.22σ | 100% |

→ **Green-List fails on the 1.1B model. SynthID does not.**

## Slide 7 — Why Green-List fails
- TinyLlama outputs are shorter and lower-entropy than GPT-2's.
- Single-key binary bias needs many diverse tokens to accumulate signal.
- Low-entropy completions dilute the intervention.
- Result: 2 of 8 watermarked texts fall below z=4.

## Slide 8 — Why SynthID survives
- 30 independent key scores, averaged per token.
- Per-key σ = 0.29 → averaged σ = 0.10.
- Tight null distribution tolerates short completions.
- **Design lesson: variance reduction beats raw bias.**

## Slide 9 — Surface attacks
- 6 transforms: identity, whitespace, lowercase, punct-strip, word-drop, word-shuffle.
- **All preserve both watermarks above z=4 on GPT-2.**
- On TinyLlama, Green-List falls on 1 of 6; SynthID holds on all 6.

## Slide 10 — Semantic attacks (the surprise)
| Attack | Green-List | SynthID |
|---|---|---|
| original | 11.08 | 14.56 |
| T5-PAWS paraphrase | 5.34 | 4.51 |
| EN-FR-EN back-translation | **7.97** | 5.09 |
| GPT-2 free rewrite | **2.09** | **0.01** |

- **Back-translation does NOT break watermarks** (contrary to folklore).
- **Only free-form deep rewriting succeeds.**

## Slide 11 — Design recommendations
1. **Small models (< 1.5B):** use SynthID-Text. Green-List's TPR is unreliable.
2. **Large models:** either scheme works for baseline calibration.
3. **Attack defense:** assume the adversary has a stronger generator; no statistical watermark survives free rewriting.
4. **Hybrid detector:** combine both tests — Green-List wins on punct-strip, SynthID wins on back-translation.

## Slide 12 — Reproducibility
- 13 Python modules, 12 CSVs, 8 figures.
- 3 git commits, deterministic seeds.
- Everything runs on a consumer CPU in under 90 minutes.
- Code: github.com/[your-handle]/greenlist-lab
