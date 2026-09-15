[![Tests](https://github.com/sidy14/greenlist-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/sidy14/greenlist-lab/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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

