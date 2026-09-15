"""make_sample.py — creates a dirty sample for testing."""
text = (
    "This is a clean sentence.\n"
    "But this\u200b line has a zero-width space.\n"
    'And this h\u0430s a Cyrillic homoglyph in place of Latin a.\n'
    "And this\u00a0line uses a non-breaking space.\n"
    "Arabic tatweel padding: \u0640\u0640\u0640\u0640\u0640 too much.\n"
)
with open("data/sample_dirty.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("wrote data/sample_dirty.txt")
