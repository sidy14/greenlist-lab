"""Final probe: inspect SynthID logits processor signatures."""
import inspect
from transformers.generation import SynthIDTextWatermarkLogitsProcessor

for name in ["__call__", "compute_g_values", "compute_ngram_keys",
             "accumulate_hash", "sample_g_values", "expected_mean_g_value"]:
    fn = getattr(SynthIDTextWatermarkLogitsProcessor, name, None)
    if fn is None:
        print(f"{name:30s} : MISSING")
        continue
    try:
        print(f"{name:30s} : {inspect.signature(fn)}")
    except Exception as e:
        print(f"{name:30s} : <no signature: {e}>")

print()
print("Properties / class attributes:")
for n in dir(SynthIDTextWatermarkLogitsProcessor):
    if not n.startswith("_") and not callable(getattr(SynthIDTextWatermarkLogitsProcessor, n, None)):
        print("  prop:", n)
