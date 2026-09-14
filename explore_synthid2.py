"""Deeper probe into SynthID detector construction."""
import inspect
from transformers import (
    SynthIDTextWatermarkDetector,
    SynthIDTextWatermarkingConfig,
)
from transformers.generation import SynthIDTextWatermarkLogitsProcessor

print("=" * 60)
print("SynthIDTextWatermarkDetector.__init__ signature")
print("=" * 60)
print(inspect.signature(SynthIDTextWatermarkDetector.__init__))

print()
print("=" * 60)
print("SynthIDTextWatermarkDetector.__call__ signature")
print("=" * 60)
try:
    print(inspect.signature(SynthIDTextWatermarkDetector.__call__))
except Exception as e:
    print("no __call__:", e)

print()
print("=" * 60)
print("SynthIDTextWatermarkLogitsProcessor.__init__ signature")
print("=" * 60)
print(inspect.signature(SynthIDTextWatermarkLogitsProcessor.__init__))

print()
print("=" * 60)
print("Where does SynthIDTextWatermarkDetector come from?")
print("=" * 60)
print(SynthIDTextWatermarkDetector.__module__)

print()
print("=" * 60)
print("Can we instantiate the detector with just a tokenizer?")
print("=" * 60)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    det = SynthIDTextWatermarkDetector(tokenizer=tok)
    print("OK:", det)
except Exception as e:
    print(type(e).__name__, ":", e)
