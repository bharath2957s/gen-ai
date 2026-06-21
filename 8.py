from langchain_core.prompts import PromptTemplate
from transformers import T5Tokenizer, T5ForConditionalGeneration

with open("document.txt", "r") as file:
    text = file.read()

template = """
Summary:
Key Points:
Conclusion:

Document:
{text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

formatted_prompt = prompt.format(text=text)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"])

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
