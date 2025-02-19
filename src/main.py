#!/usr/bin/env python
"""
main.py

A robust retrieval-augmented generation (RAG) pipeline with sophisticated embedding, candidate generation,
and reranking. This script reads partitions from a JSON file, builds a FAISS index using embeddings,
retrieves context based on a user query, generates multiple candidate summaries, and uses a cross-encoder
to select the best answer.

Usage:
    python main.py --file_path <path_to_json> --query "Your query here" [--top_k 5] [--model_name Qwen/Qwen2.5-7B-Instruct-1M]
"""

import json
import argparse
import logging
import sys

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # Use causal LM for Qwen


def read_partitions(file_path):
    """
    Reads partitions from a JSON file and returns a list of partitions with titles and content.
    
    :param file_path: Path to the JSON file.
    :return: List of dictionaries containing partition titles and text content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
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


def build_faiss_index(embeddings):
    """
    Builds a FAISS index for the provided embeddings.
    
    :param embeddings: A numpy array of shape (N, D) where N is the number of partitions and D is the embedding dimension.
    :return: A FAISS index object.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    logging.info("FAISS index built with %d vectors.", embeddings.shape[0])
    return index


def retrieve_context(query_embedding, index, partitions, top_k):
    """
    Searches the FAISS index for the nearest neighbor partitions given a query embedding.
    
    :param query_embedding: The embedding vector for the query.
    :param index: FAISS index.
    :param partitions: List of partition dictionaries.
    :param top_k: Number of top partitions to retrieve.
    :return: List of retrieved partition dictionaries.
    """
    query_embedding = np.array(query_embedding).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    logging.info("FAISS search returned indices: %s", indices)
    logging.info("FAISS search returned distances: %s", distances)
    
    retrieved_partitions = [partitions[i] for i in indices[0] if i < len(partitions)]
    logging.info("Retrieved %d partitions from FAISS.", len(retrieved_partitions))
    return retrieved_partitions


def rerank_partitions(query, partitions, cross_encoder, top_k):
    """
    Reranks retrieved partitions using a cross-encoder.
    
    :param query: The original user query.
    :param partitions: List of partition dictionaries.
    :param cross_encoder: Cross-encoder model to score relevance.
    :param top_k: Number of top partitions to select.
    :return: List of the top_k partition dictionaries sorted by relevance.
    """
    # Prepare input pairs: (query, partition text)
    encoder_inputs = [(query, part['text']) for part in partitions]
    scores = cross_encoder.predict(encoder_inputs)
    
    # Pair each partition with its score and sort in descending order
    ranked = sorted(zip(partitions, scores), key=lambda x: x[1], reverse=True)
    logging.info("Cross-encoder partition scores: %s", scores)
    
    # Log each partition's titles and its corresponding score
    for idx, (part, score) in enumerate(ranked):
        section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
        logging.info("Partition %d: Score = %.4f, Titles = %s", idx, score, section_names)
    
    # Select the top_k partitions
    top_partitions = [part for part, _ in ranked[:top_k]]
    logging.info("Selected top %d partitions after reranking.", len(top_partitions))
    return top_partitions


def generate_and_rerank(prompt, tokenizer, model, device, cross_encoder):
    """
    Generates multiple candidate answers using the Qwen model with a chat template,
    and reranks them using a cross-encoder to select the best answer.
    """
    # Wrap the prompt in Qwen's chat format
    messages = [
        {"role": "system", "content": """You are an expert M&A lawyer and investment banker specializing in analyzing M&A transactions. 
            Your task is to identify and extract key information about the M&A transaction structure, including the parties involved, 
            their roles, the basic deal terms, and the form of the transaction. You understand how companies conduct sales processes, including market checks,
            auctions, and negotiations. You can identify key events in the deal timeline, evaluate the competitiveness of a sales process, and assess
            the effectiveness of price negotiations. You are particularly skilled at analyzing board decision-making processes, the role of financial advisors,
            and governance mechanisms like special committees and majority-of-minority votes. You understand the significance of independent directors,
            controlling shareholders, and various voting requirements.  You understand how to identify management conflicts, side payments, continued employment 
            arrangements, and other potential conflicts that can arise in M&A transactions. You also are an expert in analyzing 
            shareholder responses to M&A transactions, including litigation, proxy contests, and activist campaigns. 
            You understand the typical grounds for shareholder challenges to M&A transactions and can identify key indicators of 
            shareholder opposition. You are skilled at analyzing disclosure documents to find information about shareholder 
            lawsuits, demands, and other forms of opposition to the transaction. You have extensive experience reading SEC filings and can quickly identify 
            the relevant information from complex legal documents. You understand corporate structures, parent-subsidiary relationships, 
            how to distinguish between different types of transaction parties, and the technical distinctions between different transaction structures 
            and their legal implications."""},
        {"role": "user", "content": prompt},
    ]
    
    # Format the messages using Qwen's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize the formatted prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate multiple candidate answers using beam search with constraints
    generation_output = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=False,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=512,         # limit the new tokens
        no_repeat_ngram_size=3,     # avoid repeating n-grams
        repetition_penalty=1.2      # penalize repeating text
    )
    
    # Decode candidate answers
    candidate_answers = []
    for seq in generation_output.sequences:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        # Optionally remove everything before 'Answer:' if present
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:", 1)[-1].strip()
        candidate_answers.append(decoded)
    
    # Cross-encode scoring for each candidate
    cross_encoder_inputs = [(prompt, candidate) for candidate in candidate_answers]
    scores = cross_encoder.predict(cross_encoder_inputs)
    
    # Log each candidate answer with its corresponding score
    for idx, (candidate, score) in enumerate(zip(candidate_answers, scores)):
        logging.info("Candidate %d: Score = %.4f, Answer = %s", idx, score, candidate)
    
    best_index = int(np.argmax(scores))
    best_answer = candidate_answers[best_index]
    logging.info("Selected candidate %d with score %.4f.", best_index, scores[best_index])
    
    return best_answer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation Pipeline with Sophisticated Reranking")
    parser.add_argument("--file_path", default='/gpfs/scratch/wz1492/rag-law/data/example_data.json', help="Path to the JSON file containing partitions.")
    parser.add_argument("--query", default='', help="A summary of the nature of the transaction at issue in this document.")
    parser.add_argument("--top_k", type=int, default=40, help="Number of partitions to retrieve and rerank.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M",
                        help="Name of the generative model to use (for Qwen, use Qwen/Qwen2.5-7B-Instruct-1M).")
    args = parser.parse_args()

    # Read partitions from the JSON file
    partitions = read_partitions(args.file_path)
    logging.info("Loaded %d partitions from file.", len(partitions))
    
    # Extract texts for embeddings while keeping full partition info for later logging.
    partition_texts = [p['text'] for p in partitions]
    
    # Initialize SentenceTransformer for computing embeddings
    logging.info("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    logging.info("Computing embeddings for partitions...")
    embeddings = embedder.encode(partition_texts, convert_to_numpy=True)
    
    # Build FAISS index for context retrieval
    index = build_faiss_index(embeddings)
    
    # Compute embedding for the query
    logging.info("Computing embedding for the query...")
    query_embedding = embedder.encode(args.query, convert_to_numpy=True)
    
    # Retrieve top_k partitions from FAISS (returns full partition data)
    logging.info("Retrieving top %d partitions relevant to the query...", args.top_k)
    retrieved_partitions = retrieve_context(query_embedding, index, partitions, args.top_k)
    
    # Load cross-encoder (used for both partition reranking and candidate answer reranking)
    logging.info("Loading cross-encoder for reranking...")
    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    except Exception as e:
        logging.error("Error loading cross-encoder: %s", e)
        sys.exit(1)
    
    # Rerank the retrieved partitions using the cross-encoder
    logging.info("Reranking retrieved partitions based on query relevance...")
    top_partitions = rerank_partitions(args.query, retrieved_partitions, cross_encoder, args.top_k)
    
    # Log the section names of the selected partitions
    for idx, part in enumerate(top_partitions):
        section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
        logging.info("Selected Partition %d Section Name(s): %s", idx, section_names)
    
    # Concatenate the top partitions' text to form the refined context
    context = "\n\n".join([part['text'] for part in top_partitions])
    logging.info("Constructed refined context from top partitions.")
    
    # Build the prompt tailored for a transaction summary
    prompt = (
        f"Context:\n{context}\n\n"
        "Task: Provide a concise summary of the nature of the transaction at issue in the document. "
        "Address key aspects such as parties involved, basic deal terms, transaction form, timeline, sales process, "
        "board decision-making, financial advisxor roles, governance mechanisms, and potential conflicts. "

    )
    logging.info("Constructed prompt for generation.")
    
    # Initialize the generative model and tokenizer using Qwen as a causal language model with chat capabilities
    logging.info("Loading generative model: %s", args.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        logging.error("Error loading the model %s: %s", args.model_name, e)
        sys.exit(1)
    
    # Move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info("Generative model moved to device: %s", device)
    
    # Generate candidate answers and select the best one via reranking
    logging.info("Generating and reranking candidate answers...")
    final_answer = generate_and_rerank(prompt, tokenizer, model, device, cross_encoder)
    
    # Output the result
    print("=== Answer ===")
    print(final_answer)


if __name__ == "__main__":
    main()
