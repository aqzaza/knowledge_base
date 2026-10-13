"""Compare Chinese and English retrieval queries on the same local index."""

import json
import argparse
from pathlib import Path

from embed_query import load_index, retrieve


def reference_hit(hits: list, references: list[dict]) -> tuple[bool, list[int]]:
    reference_keys = {
        (item["source"], item["start_index"])
        for item in references
    }
    matched_ranks = [
        rank
        for rank, (_, chunk, _) in enumerate(hits, start=1)
        if (chunk.metadata["source"], chunk.metadata["start_index"]) in reference_keys
    ]
    return bool(matched_ranks), matched_ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model",
        choices=("multilingual-e5-small", "multilingual-e5-base", "bge-m3"),
        default="multilingual-e5-base",
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent
    questions = json.loads(
        (project_dir / "eval" / "dev_questions.json").read_text(encoding="utf-8")
    )
    model, chunks, passage_vectors, use_e5_prefixes = load_index(args.embedding_model)
    print("Embedding model:", args.embedding_model)

    for item in questions:
        print("\n" + "=" * 72)
        print(item["id"], item["question"])
        for language, query in (
            ("Chinese", item["question"]),
            ("English", item["retrieval_question"]),
        ):
            hits = retrieve(query, model, chunks, passage_vectors, use_e5_prefixes, k=5)
            hit, ranks = reference_hit(hits, item["reference_evidence"])
            print(f"\n{language} query: {query}")
            print("Reference in Top-5:", hit, "Ranks:", ranks)
            for rank, (index, chunk, score) in enumerate(hits, start=1):
                print(
                    f"  Rank {rank}: chunk={index}, score={score:.4f}, "
                    f"source={chunk.metadata['source']}, "
                    f"start={chunk.metadata['start_index']}"
                )


if __name__ == "__main__":
    main()
