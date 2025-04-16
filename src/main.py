#!/usr/bin/env python
"""
main.py

A robust retrieval-augmented retrieval pipeline wrapped in a class named LLMReRankExtractor.
Reranking is always enabled by default (no property or toggle for disabling it).

This class:
  1. Reads partitions from a JSON file.
  2. Reranks *all* partitions using a cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-12-v2, configurable via --model_name).
  3. Prints out the full ranking of all partitions.
  4. Selects the top_k partitions to form the concatenated final answer.
  5. Saves the output for each query to a text file, showing the final answer and ranking details.
  6. Runs fixed queries by default if no queries are provided.

Usage:
    python main.py --file_path <path_to_json> [--top_k 100] [--model_name <cross_encoder_model_name>] [--api_key <api_key>]
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from sentence_transformers import CrossEncoder

# Set the XDG_CONFIG_HOME to a local directory
local_config = os.path.join(os.getcwd(), ".config")
os.makedirs(local_config, exist_ok=True)
os.environ["XDG_CONFIG_HOME"] = local_config


class LLMReRankExtractor:
    def __init__(self, file_path, top_k=20, model_name=None, api_key=None):
        """
        Initialize the LLMReRankExtractor.

        Args:
            file_path (str): Path to the JSON file with partitions.
            top_k (int, optional): Number of partitions to select from the full ranked list. Defaults to 20.
            model_name (str, optional): Model name for the cross-encoder. Defaults to 'cross-encoder/ms-marco-MiniLM-L-12-v2'.
            api_key (str, optional): API key for any external service (if needed). Defaults to None.
        """
        self.file_path = file_path
        self.top_k = top_k
        self.model_name = model_name if model_name else "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.api_key = api_key
        self.partitions = None

        logging.info(f"Cross-encoder model set to: {self.model_name}")
        logging.info(f"API key provided: {'Yes' if self.api_key else 'No'}")

        # Load cross-encoder for reranking
        logging.info(f"Loading cross-encoder ({self.model_name})...")
        try:
            self.cross_encoder = CrossEncoder(self.model_name)
        except Exception as e:
            logging.error("Error loading cross-encoder '%s': %s", self.model_name, e)
            sys.exit(1)

    def _load_partitions(self):
        """Loads and returns partition data from the JSON file."""
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

        return [
            {
                "titles": partition.get("titles", []),
                "text": partition.get("text", "")
            }
            for partition in partitions
        ]

    def setup_pipeline(self):
        """Loads partitions."""
        logging.info("Loading partitions from file: %s", self.file_path)
        self.partitions = self._load_partitions()
        logging.info("Loaded %d partitions.", len(self.partitions))

    def _rerank_partitions(self, query, partitions):
        """
        Rerank partitions using a cross-encoder. Includes titles in the text for ranking.
        Returns the sorted list (descending by score).
        """
        # Combine titles and text for each partition before ranking
        encoder_inputs = []
        for part in partitions:
            title_str = ", ".join(part.get('titles', []))
            combined_text = f"Titles: {title_str}\n\n{part.get('text', '')}"
            encoder_inputs.append((query, combined_text))

        scores = self.cross_encoder.predict(encoder_inputs)

        # zip each partition with its cross-encoder score
        partition_score_pairs = list(zip(partitions, scores))

        # sort them in descending order of cross-encoder score
        ranked = sorted(partition_score_pairs, key=lambda x: x[1], reverse=True)
        return ranked

    def process_query(self, query_label, query_text):
        """
        Processes a single query by reranking all partitions,
        printing the full ranking, and then selecting the top_k for the final context.
        """
        logging.info("Processing query %s: %s", query_label, query_text)

        # 1) Rerank all loaded partitions (using cross-encoder scores)
        if not self.partitions:
            logging.error("Partitions not loaded. Run setup_pipeline() first.")
            return None

        ranked_pairs = self._rerank_partitions(query_text, self.partitions)
        logging.info("Reranked all %d partitions for query: '%s'", len(self.partitions), query_text)

        # Print out the entire ranking
        # Each element in ranked_pairs is (partition, score)
        logging.info("=== Full Ranking of Partitions (Descending Cross-Encoder Score) ===")
        for idx, (part, score) in enumerate(ranked_pairs):
            section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
            logging.info("Rank %d: Score = %.4f | Titles = %s", idx+1, score, section_names)

        # 2) Select the top_k from the reranked list
        top_partitions = [x[0] for x in ranked_pairs[:self.top_k]]
        top_scores = [x[1] for x in ranked_pairs[:self.top_k]]

        # 3) Concatenate the top_k texts, including titles
        context_parts = []
        for p in top_partitions:
            title_str = ", ".join(p.get('titles', []))
            combined_text = f"Titles: {title_str}\n\n{p.get('text', '')}"
            context_parts.append(combined_text)

        context = "\n\n---\n\n".join(context_parts) # Use a separator between combined partition texts
        logging.info("Constructed refined context from top %d partitions (including titles).", self.top_k)

        # 4) That context is our final answer
        final_answer = context
        logging.info("Returning retrieved context as the final answer for query %s.", query_label)

        # Prepare the output text, including the final answer and the entire ranking
        output_text = f"=== Answer for Query {query_label} ({query_text}) ===\n\n{final_answer}\n"

        output_text += "\n=== Full Ranking (Cross-Encoder Scores) ===\n"
        for idx, (part, score) in enumerate(ranked_pairs):
            section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
            output_text += f"Rank {idx+1}: {section_names} | Score: {score:.4f}\n"

        output_filename = f"output_{query_label}.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as outfile:
                outfile.write(output_text)
            logging.info("Saved generated output for query %s to %s", query_label, output_filename)
        except Exception as e:
            logging.error("Error writing output for query %s: %s", query_label, e)

        # Also print the result to stdout
        print(output_text)
        print("\n" + "=" * 40 + "\n")

        return output_text

    def run_pipeline(self, queries=None):
        """
        Run the complete pipeline for a set of queries.
        If no queries are provided, a default set of descriptive queries is used.
        """
        if queries is None:
            queries = [
                (
                    "A",
                    "Locate all sections that contain tables setting forth the information required by Item 402(t) of Regulation S-K promulgated under the Exchange Act regarding the compensation that is or may become payable to a company's named executive officers that is based on or otherwise relates to a merger or acquisition. This compensation is referred to as \"golden parachute\" compensation by applicable SEC disclosure rules"
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
        description="Rerank Pipeline using LLMReRankExtractor, reranking all partitions using a CrossEncoder and selecting top_k."
    )
    parser.add_argument("--file_path", default='data/0000832428-15-000023.json', help="Path to the JSON file containing partitions.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of partitions to select from the final reranked list.")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model to use for the cross-encoder. Defaults to cross-encoder/ms-marco-MiniLM-L-12-v2.")
    parser.add_argument("--api_key", type=str, default=None, help="API key if needed for external services.")
    args = parser.parse_args()

    extractor = LLMReRankExtractor(
        file_path=args.file_path,
        top_k=args.top_k,
        model_name=args.model_name,
        api_key=args.api_key
    )
    extractor.setup_pipeline()
    extractor.run_pipeline()
