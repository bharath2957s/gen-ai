from transformers import pipeline

summarizer = pipeline("summarization")

text = """
Artificial Intelligence is transforming industries. 
It helps in automation, healthcare, education, and robotics. 
AI improves efficiency and decision-making.
"""

summary = summarizer(text)

print(summary)
