from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

print(len(splits))

## Embeddings
# Let's take our splits and embed them.

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

print(np.dot(embedding1, embedding2))
print(np.dot(embedding1, embedding3))
print(np.dot(embedding2, embedding3))

## Vectorstores
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.vectorstores import Chroma

persist_directory = "docs/chroma"

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

## Similarity Search
question = "is there an email i can ask for help"
docs = vectordb.similarity_search(question, k=3)

print(len(docs))

print(docs[0].page_content)
print("")
print(docs[1].page_content)
print("")
print(docs[2].page_content)

#Let's save this so we can use it later!
vectordb.persist()

## Failure modes
# This seems great, and basic similarity search will get you 80% of the way there very easily.
# But there are some failure modes that can creep up.
# Here are some edge cases that can arise - we'll fix them in the next class.

question = "what did they say about matlab?"
docs = vectordb.similarity_search(question, k=5)

print(docs[0])
print(docs[1])

# We can see a new failure mode.
# The question below asks a question about the third lecture, but includes results from other lectures as well.

question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question, k=5)

for doc in docs:
    print(doc.metadata)

print(docs[4].page_content)