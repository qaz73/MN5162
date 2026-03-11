
from transformers import pipeline
import torch
print(f'\n MPS available { torch.backends.mps.is_available()}')

generator = pipeline("text-generation", model="gpt2")

result = generator("Hello world, today is", max_length=30)

generated_text = result[0]["generated_text"]

summarizer = pipeline("summarization")

summary = summarizer(generated_text, max_length=40, min_length=10, do_sample=False)

print(f'Generated Text {generated_text}')

print(f'Summary {summary}')


