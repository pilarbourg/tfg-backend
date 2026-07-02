import json
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

with open("/Users/pilarbourg/Desktop/tfg-pipeline/test/ragas/limited_rag_results.json") as f:
    samples = json.load(f)

judge = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    temperature=0.0,
)

embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/S-PubMedBert-MS-MARCO")

run_config = RunConfig(timeout=600, max_retries=3, max_wait=120, max_workers=1)

result = evaluate(
    dataset=EvaluationDataset.from_list(samples),
    metrics=[ResponseRelevancy()],
    #  ContextPrecision(), ContextRecall(), Faithfulness(), ResponseRelevancy()
    llm=LangchainLLMWrapper(judge),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
    run_config=run_config,
)

result.to_pandas().to_csv("limited_ragas_results.csv", index=False)