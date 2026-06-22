from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import T5Tokenizer, T5ForConditionalGeneration
loader = PyPDFLoader("ipc.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(texts, embedding)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
query = input("Ask Question: ")
docs = db.similarity_search(query)
context = docs[0].page_content
prompt = f"Answer based on IPC:\n{context}\nQuestion:{query}"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"])
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
