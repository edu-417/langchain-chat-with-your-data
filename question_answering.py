## LLM to use LLAMA2 7B CHAT from HuggingFace
from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    # trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id
)

pipeline = transformers.pipeline(
    task='text-generation',
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    # we pass model parameters here too
    # temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    # max_new_tokens=512,  # max number of tokens to generate in the output
    # repetition_penalty=1.1  # without this output begins repeating
)

### Question Answering

from langchain.vectorstores import Chroma
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.embeddings import HuggingFaceEmbeddings

persist_directory = 'docs/chroma/'
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())

question = "What are major topics for this class?"
# question = "¿Hay algun prerequisito para esta clase?"
docs = vectordb.similarity_search(question,k=3)
print(len(docs))
# print(docs)

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(
    pipeline=pipeline,
    pipeline_kwargs={
        "temperature": 0,
        "min_new_tokens": 64,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id
    }
)

## RetrievalQA chain

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain({"query": question})
print(result)

## Prompt

from langchain.prompts import PromptTemplate

template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "Is probability a class topic?"
result = qa_chain({"query": question})

print(result["result"])

print(result["source_documents"][0])

## RetrievalQA chain types

qa_chain_mr = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

result = qa_chain_mr({"query": question})

print(result)