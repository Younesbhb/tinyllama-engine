from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tests = [
    "Hello",
    " Hello", 
    "Hello world",
    "The quick brown fox",
    "123",
    "don't",
]

for text in tests:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids)
    print(f"'{text}' → {ids} → '{decoded}'")