from transformers import pipeline
print("loading modules")
summarizer = pipeline("summarization")
print("Model loaded successfully.\n")
text = """
Artificial Intelligence is transforming industries. 
It helps in automation, healthcare, education, and robotics. 
AI improves efficiency and decision-making.
"""
summary = summarizer(text)
print(summary)

