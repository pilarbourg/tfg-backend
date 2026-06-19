import json
import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.rag.retrieval import perform_search_with_rerank_hybrid
from app.core.rag.generation import get_atlas_stream


def run_pipeline(query):
    chunks = perform_search_with_rerank_hybrid(query)
    context_text = "\n".join(f"DOI: {c['meta']['url']}\nContent: {c['text']}" for c in chunks)
    answer = "".join(c.content for c in get_atlas_stream(query, context_text))
    return answer, [c["text"] for c in chunks]


with open("validation_dataset.json") as f:
    questions = json.load(f)["questions"]

samples = []
for q in questions:
    answer, contexts = run_pipeline(q["sample_query"])
    samples.append({
        "user_input": q["sample_query"],
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": q["expected_output"],
    })

judge = ChatOllama(model="deepseek-r1:14b", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

result = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), ResponseRelevancy()],
    llm=LangchainLLMWrapper(judge),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)

print(result)
result.to_pandas().to_csv("ragas_results.csv", index=False)