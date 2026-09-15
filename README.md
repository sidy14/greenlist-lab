[![Tests](https://github.com/sidy14/greenlist-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/sidy14/greenlist-lab/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Green-List vs SynthID-Text: A Statistical Watermark Study

Reproducible study of the Kirchenbauer et al. (2023) green-list watermark
and Google DeepMind''s SynthID-Text (2024), with a unified Python toolkit
for detecting and cleaning AI-generated text artifacts in Latin and Arabic.

## Headline result

| Model | Params | Vocab | Green-List TPR | SynthID TPR |
|---|---|---|---|---|
| GPT-2 (base) | 124M | 50,257 | 100% | 100% |
| Qwen 2.5 (chat) | 500M | 151,665 | 0% | 100% |
| TinyLlama (chat) | 1.1B | 32,000 | 75% | 100% |

Finding: Green-List fails above ~100k vocabulary size. SynthID-Text is
immune thanks to its 30-key averaging.

## Repository layout

src/            28 Python modules - toolkit + research scripts
tests/          59 unit tests
data/reports/   CSVs + 11 publication figures
paper/          LaTeX source, compiled PDF, slide deck
dev/            Scratch scripts used during exploration

## Install

git clone https://github.com/sidy14/greenlist-lab.git
cd greenlist-lab
pip install -e .[dev]

Lightweight install (no torch required for text cleaning):
pip install -e . --no-deps
pip install numpy

## Use the toolkit

ai-text-lab analyze file.txt --model openai-community/gpt2
ai-text-lab clean file.txt -o cleaned.txt
ai-text-lab diff file.txt

import ai_text_lab
report = ai_text_lab.analyze(text, model_id=''openai-community/gpt2'')
print(report.to_human())
cleaned = ai_text_lab.clean(text)


## 4-layer AI-text detection

\\ash
ai-text-lab detect file.txt        # full (vendor + surface + stylometric + Arabic)
ai-text-lab detect file.txt --fast # skip stylometric (no torch needed)
ai-text-lab detect file.txt --json # machine-readable
\
Detects signatures from:
- **OpenAI o3/o4-mini** — deterministic (NNBSP U+202F markers)
- **Anthropic Claude** — thinking tags, system prompt leakage
- **Google Gemini** — control tokens
- **Meta Llama 3** — chat template markers
- **Statistical watermarks** — Green-List, SynthID (with tokenizer)
- **Stylometric features** — perplexity, burstiness, lexical diversity
- **Arabic-specific AI markers** — formal phrases, tashkeel patterns

## Streamlit UI

\\ash
pip install streamlit
streamlit run app.py
\
Opens a bilingual (English / Arabic) interface at http://localhost:8501

## Reproduce the paper

python -m pytest tests/ -v
python src/run_calibration.py
python src/tinyllama_all.py
python src/qwen_all.py
python src/plot_three_models.py

## Citation

@misc{cissokho2026watermarks,
  title  = {When Watermarks Fail: Non-Monotonic Sensitivity of Statistical LLM Watermarks to Model Size and Vocabulary},
  author = {Cissokho, Sidy},
  year   = {2026},
  url    = {https://github.com/sidy14/greenlist-lab}
}

## License

MIT - see LICENSE.
