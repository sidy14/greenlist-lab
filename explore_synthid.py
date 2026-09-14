"""Explore the SynthID-Text API surface."""
import inspect
from transformers import (
    SynthIDTextWatermarkDetector,
    SynthIDTextWatermarkingConfig,
)
from transformers.generation import SynthIDTextWatermarkLogitsProcessor

print("=== SynthIDTextWatermarkingConfig ===")
for n, p in inspect.signature(SynthIDTextWatermarkingConfig.__init__).parameters.items():
    if n != "self":
        print(f"  {n:25s} default={p.default!r}")

print()
print("=== SynthIDTextWatermarkDetector ===")
for n, p in inspect.signature(SynthIDTextWatermarkDetector.__init__).parameters.items():
    if n != "self":
        print(f"  {n:25s} default={p.default!r}")

print()
print("=== SynthIDTextWatermarkLogitsProcessor methods ===")
print([m for m in dir(SynthIDTextWatermarkLogitsProcessor) if not m.startswith("_")])
