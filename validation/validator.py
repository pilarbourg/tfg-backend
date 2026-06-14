import sys
import types

mock_vertex_module = types.ModuleType("langchain_community.chat_models.vertexai")

class FakeChatVertexAI:
    pass

mock_vertex_module.ChatVertexAI = FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_module

import os
import json
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.app.core.rag.retrieval import get_context
from src.app.core.rag.generation import get_atlas_stream

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUND_TRUTH_PATH = os.path.join(PROJECT_ROOT, "validation", "data", "ground_truth.json")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "validation", "data")
RESULTS_PATH = os.path.join(RESULTS_DIR, "eval_results.json")

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "pipeline_checkpoint.json")

MAX_QA_PAIRS = None

OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def load_validation_set(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        raise ValueError(f"No 'qa_pairs' found in {path}. Check the file structure.")

    validation_set = []
    for record in qa_pairs:
        validation_set.append({
            "question": record["question"],
            "ground_truth": record["ground_truth_answer"],
        })

    return validation_set


async def run_rag_pipeline(user_query: str):
    """
    Simulates a live run of your pipeline to gather all components
    required by the Ragas evaluation framework.
    """
    context_chunks, _ = get_context(user_query, target_table="validation_papers")
    retrieved_chunks = [chunk['text'] for chunk in context_chunks]

    stream_response = get_atlas_stream(user_query, retrieved_chunks)

    generated_answer = ""
    for chunk in stream_response:
        text_piece = getattr(chunk, "content", getattr(chunk, "text", None))
        if text_piece:
            generated_answer += text_piece

    return {
        "retrieved_contexts": retrieved_chunks,
        "response": generated_answer
    }


async def main():
    load_dotenv()

    print(f"Loading validation set from {GROUND_TRUTH_PATH}...")
    validation_set = load_validation_set(GROUND_TRUTH_PATH)
    print(f"   Loaded {len(validation_set)} QA pairs.")

    if MAX_QA_PAIRS is not None:
        validation_set = validation_set[:MAX_QA_PAIRS]
        print(f"   ⚠️  Limiting this run to {len(validation_set)} QA pair(s) "
              f"(MAX_QA_PAIRS={MAX_QA_PAIRS}).")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    eval_records = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            eval_records = json.load(f)
        print(f"♻️  Resuming from checkpoint: {len(eval_records)} record(s) already generated.")

    done_questions = {rec["question"] for rec in eval_records}

    print("🚀 Running pipeline over evaluation samples...")
    for record in validation_set:
        query = record["question"]

        if query in done_questions:
            print(f"Skipping already-generated question: {query[:60]}...")
            continue

        pipeline_output = await run_rag_pipeline(query)

        eval_records.append({
            "question": query,
            "contexts": pipeline_output["retrieved_contexts"],
            "answer": pipeline_output["response"],
            "ground_truth": record["ground_truth"]
        })

        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(eval_records, f, indent=4, ensure_ascii=False)
        print(f"Checkpoint saved ({len(eval_records)} record(s)).")

    eval_dataset = Dataset.from_list(eval_records)

    print(f"Evaluating dataset with Ragas metrics using local Ollama "
          f"({OLLAMA_JUDGE_MODEL})...")

    ollama_model = ChatOllama(
        model=OLLAMA_JUDGE_MODEL,
        temperature=0.0,
    )
    judge_llm = LangchainLLMWrapper(ollama_model)

    ollama_embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    judge_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

    run_config = RunConfig(max_workers=1, timeout=600)

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                LLMContextPrecisionWithReference(),
                LLMContextRecall(),
            ],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print(f"   Your generated answers are safe in {CHECKPOINT_PATH}.")
        print("   Re-run — generation will be skipped and it will retry evaluation directly.")
        raise

    print("\n✅ Evaluation complete! System Results:")
    print(result)

    df_results = result.to_pandas()
    df_results.to_json(RESULTS_PATH, orient="records", indent=4)
    print(f"Metrics successfully exported to {RESULTS_PATH}")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("🧹 Cleared generation checkpoint.")


if __name__ == "__main__":
    asyncio.run(main())