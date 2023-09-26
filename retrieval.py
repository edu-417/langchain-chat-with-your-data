from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#### Retrieval
# Retrieval is the centerpiece of our retrieval augmented generation (RAG) flow.

# Let's get our vectorDB from before.

### Vectorstore retrieval

## Similarity Search

persist_directory = "docs/chroma"

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"

print("SIMILARITY SEARCH:")
print(smalldb.similarity_search(question, k=2))

print("MAX MARGINAL RELEVANCE SEARCH")
print(smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3))


### Addressing Diversity: Maximum marginal relevance
# Last class we introduced one problem: how to enforce diversity in the search results.
# Maximum marginal relevance strives to achieve both relevance to the query and diversity among the results.

question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question, k=3)

print("")
print(docs_ss[0].page_content)
print("")
print(docs_ss[1].page_content)

