#!/usr/bin/env python

"""
main.py

A robust retrieval-augmented retrieval pipeline wrapped in a class named RerankExtractor.
Reranking is always enabled by default (no property or toggle for disabling it).

This class:
  1. Reads partitions from a JSON file.
  2. Builds a FAISS index using embeddings (default model is all-MiniLM-L6-v2, can be overridden by --model_name).
  3. Retrieves context based on a retrieval query.
  4. Reranks the retrieved context based on the retrieval query using a cross-encoder (always uses cross-encoder/ms-marco-MiniLM-L-12-v2).
  5. Returns the concatenated retrieved context as the answer.
  6. Reports section scores for each partition.
  7. Runs fixed queries and saves each output in a text file.
     
Usage:
    python main.py --file_path <path_to_json> [--top_k 100] [--model_name <model_name>] [--api_key <api_key>]
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Set the XDG_CONFIG_HOME to a local directory
local_config = os.path.join(os.getcwd(), ".config")
os.makedirs(local_config, exist_ok=True)
os.environ["XDG_CONFIG_HOME"] = local_config

class RerankExtractor:
    def __init__(self, file_path, top_k=100, model_name=None, api_key=None):
        """
        Initialize the RerankExtractor.

        Args:
            file_path (str): Path to the JSON file with partitions.
            top_k (int, optional): Number of partitions to retrieve. Defaults to 100.
            model_name (str, optional): Model name for the embedding model. Defaults to 'all-MiniLM-L6-v2'.
            api_key (str, optional): API key for any external service (if needed). Defaults to None.
        """
        self.file_path = file_path
        self.top_k = top_k
        self.model_name = model_name if model_name else "all-MiniLM-L6-v2"
        self.api_key = api_key
        self.partitions = None
        self.index = None

        # Log model_name and api_key if provided
        logging.info(f"Embedding model set to: {self.model_name}")
        logging.info(f"API key provided: {'Yes' if self.api_key else 'No'}")

        # Load embedding model (for partitions and queries) using model_name
        logging.info("Loading embedding model...")
        try:
            self.embedder = SentenceTransformer(self.model_name)
        except Exception as e:
            logging.error("Error loading embedding model '%s': %s", self.model_name, e)
            sys.exit(1)

        # Load cross-encoder for reranking (fixed model)
        logging.info("Loading cross-encoder (cross-encoder/ms-marco-MiniLM-L-12-v2) for reranking...")
        try:
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        except Exception as e:
            logging.error("Error loading cross-encoder: %s", e)
            sys.exit(1)

    def _load_partitions(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.error("Failed to load JSON file %s: %s", self.file_path, e)
            sys.exit(1)
        partitions = data.get("partitions", [])
        if not partitions:
            logging.error("No partitions found in the JSON file.")
            sys.exit(1)
        return [{"titles": partition.get("titles", []), "text": partition.get("text", "")} for partition in partitions]

    def _build_faiss_index(self, embeddings):
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        logging.info("FAISS index built with %d vectors.", embeddings.shape[0])
        return index

    def setup_pipeline(self):
        """Load partitions, compute embeddings, and build the FAISS index."""
        logging.info("Loading partitions from file: %s", self.file_path)
        self.partitions = self._load_partitions()
        logging.info("Loaded %d partitions.", len(self.partitions))
        partition_texts = [p['text'] for p in self.partitions]
        logging.info("Computing embeddings for partitions...")
        embeddings = self.embedder.encode(partition_texts, convert_to_numpy=True)
        self.index = self._build_faiss_index(embeddings)

    def _retrieve_context(self, query_embedding):
        """
        Retrieves partitions using FAISS.
        Returns:
            retrieved_partitions: list of partition dictionaries,
            scores: corresponding FAISS distance scores,
            indices: corresponding indices from FAISS.
        """
        query_embedding = np.array(query_embedding).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, self.top_k)
        logging.info("FAISS search returned indices: %s", indices)
        logging.info("FAISS search returned distances: %s", scores)
        retrieved_partitions = [self.partitions[i] for i in indices[0] if i < len(self.partitions)]
        return retrieved_partitions, scores[0], indices[0]

    def _rerank_partitions(self, query, partitions):
        """
        Rerank partitions using a cross-encoder.
        Returns:
            top_partitions: list of top partitions (up to top_k),
            ranked: full ranking list as tuples (partition, score).
        """
        encoder_inputs = [(query, part['text']) for part in partitions]
        scores = self.cross_encoder.predict(encoder_inputs)
        ranked = sorted(zip(partitions, scores), key=lambda x: x[1], reverse=True)
        logging.info("Cross-encoder partition scores: %s", scores)
        for idx, (part, score) in enumerate(ranked):
            section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
            logging.info("Partition %d: Score = %.4f, Titles = %s", idx, score, section_names)
        top_partitions = [part for part, _ in ranked[:self.top_k]]
        logging.info("Selected top %d partitions after reranking.", len(top_partitions))
        return top_partitions, ranked

    def process_query(self, query_label, query_text):
        logging.info("Processing query %s: %s", query_label, query_text)
        # Compute embedding for the query text
        retrieval_query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)
        logging.info("Retrieving top %d partitions relevant to the query...", self.top_k)
        retrieved_partitions, faiss_scores, _ = self._retrieve_context(retrieval_query_embedding)

        # Always rerank the retrieved partitions
        logging.info("Reranking retrieved partitions for query: %s", query_text)
        top_partitions, reranked_details = self._rerank_partitions(query_text, retrieved_partitions)
        score_header = "Reranked Partitions and Cross-Encoder Scores"

        for idx, part in enumerate(top_partitions):
            section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
            logging.info("Selected Partition %d Section Name(s): %s", idx, section_names)

        # Concatenate texts from top partitions to form the refined context
        context = "\n\n".join([part['text'] for part in top_partitions])
        logging.info("Constructed refined context from top partitions.")

        # Since the generation model has been removed, we directly use the retrieved context as the answer.
        final_answer = context
        logging.info("Returning retrieved context as answer for query %s.", query_label)

        # Prepare output text including answer and section scores
        output_text = f"=== Answer for Query {query_label} ({query_text}) ===\n\n{final_answer}\n"
        output_text += f"\n=== {score_header} ===\n"
        for idx, (part, score) in enumerate(reranked_details):
            section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
            output_text += f"Rank {idx+1}: {section_names} | Score: {score:.4f}\n"

        output_filename = f"output_{query_label}.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as outfile:
                outfile.write(output_text)
            logging.info("Saved generated output for query %s to %s", query_label, output_filename)
        except Exception as e:
            logging.error("Error writing output for query %s: %s", query_label, e)

        print(output_text)
        print("\n" + "=" * 40 + "\n")
        return output_text

    def run_pipeline(self, queries=None):
        """
        Run the complete pipeline for a set of queries. If no queries are provided,
        a default set of descriptive queries is used.
        """
        if queries is None:
            queries = [
                (
                    "A",
                    "Section discussing executive severance payments, termination compensation, or financial arrangements for the CEO in the event of a merger, acquisition, or company sale. Includes details on cash payouts, stock options, benefits, or bonuses triggered by a change of control."
                ),
            ]
        outputs = {}
        for label, query_text in queries:
            output_text = self.process_query(label, query_text)
            outputs[label] = output_text
        return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Rerank Pipeline using RerankExtractor."
    )
    parser.add_argument("--file_path", default='data/example_sec.json', help="Path to the JSON file containing partitions.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of partitions to retrieve and rerank.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to use for the embedding model. Defaults to all-MiniLM-L6-v2.")
    parser.add_argument("--api_key", type=str, default=None, help="API key if needed for external services.")
    args = parser.parse_args()

    extractor = RerankExtractor(
        file_path=args.file_path,
        top_k=args.top_k,
        model_name=args.model_name,
        api_key=args.api_key
    )
    extractor.setup_pipeline()
    extractor.run_pipeline()



