#!/usr/bin/env python
"""
main.py

A robust retrieval-augmented generation (RAG) pipeline that:
1. Reads partitions from a JSON file.
2. Builds a FAISS index using embeddings.
3. Retrieves context based on a user query.
4. Optionally reranks context based on a user query if --rerank is enabled.
5. Generates multiple candidate summaries using Qwen via vllm (following its official prompting style).
6. Optionally uses a cross-encoder to select the best answer if --rerank is enabled.

Usage:
    python main.py --file_path <path_to_json> --query "Your query here" [--top_k 5] [--model_name Qwen/Qwen2.5-7B-Instruct-1M] [--rerank]
"""
import os

# Set the XDG_CONFIG_HOME to a local directory (e.g., "./.config")
local_config = os.path.join(os.getcwd(), ".config")
os.makedirs(local_config, exist_ok=True)
os.environ["XDG_CONFIG_HOME"] = local_config

import json
import argparse
import logging
import sys
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def read_partitions(file_path):
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
    return [{"titles": partition.get("titles", []), "text": partition.get("text", "")} for partition in partitions]

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    logging.info("FAISS index built with %d vectors.", embeddings.shape[0])
    return index

def retrieve_context(query_embedding, index, partitions, top_k):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    logging.info("FAISS search returned indices: %s", indices)
    logging.info("FAISS search returned distances: %s", distances)
    retrieved_partitions = [partitions[i] for i in indices[0] if i < len(partitions)]
    logging.info("Retrieved %d partitions from FAISS.", len(retrieved_partitions))
    return retrieved_partitions

def rerank_partitions(query, partitions, cross_encoder, top_k):
    # Prepare input pairs: (query, partition text)
    encoder_inputs = [(query, part['text']) for part in partitions]
    scores = cross_encoder.predict(encoder_inputs)
    ranked = sorted(zip(partitions, scores), key=lambda x: x[1], reverse=True)
    logging.info("Cross-encoder partition scores: %s", scores)
    for idx, (part, score) in enumerate(ranked):
        section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
        logging.info("Partition %d: Score = %.4f, Titles = %s", idx, score, section_names)
    top_partitions = [part for part, _ in ranked[:top_k]]
    logging.info("Selected top %d partitions after reranking.", len(top_partitions))
    return top_partitions

def generate_and_rerank(prompt, tokenizer, llm, sampling_params, cross_encoder, num_candidates=5):
    # Create messages using Qwen's expected chat format
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    # Format the prompt using Qwen's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Replicate the prompt to generate multiple  answers
    prompts = [formatted_prompt] * num_candidates
    outputs = llm.generate(prompts, sampling_params)
    
    candidate_answers = []
    for output in outputs:
        generated_text = output.outputs[0].text
        # If the generation follows a "Answer:" structure, extract the part after it.
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:", 1)[-1].strip()
        candidate_answers.append(generated_text)
    
    # Rerank candidate answers using the cross-encoder
    cross_encoder_inputs = [(prompt, candidate) for candidate in candidate_answers]
    scores = cross_encoder.predict(cross_encoder_inputs)
    
    for idx, (candidate, score) in enumerate(zip(candidate_answers, scores)):
        logging.info("Candidate %d: Score = %.4f, Answer = %s", idx, score, candidate)
    
    best_index = int(np.argmax(scores))
    best_answer = candidate_answers[best_index]
    logging.info("Selected candidate %d with score %.4f.", best_index, scores[best_index])
    
    return best_answer

def generate_without_rerank(prompt, tokenizer, llm, sampling_params):
    # Simple generation without candidate reranking
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate([formatted_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    if "Answer:" in generated_text:
        generated_text = generated_text.split("Answer:", 1)[-1].strip()
    return generated_text

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation Pipeline with vllm-based Qwen Inference")
    parser.add_argument("--file_path", default='data/example_data.json', help="Path to the JSON file containing partitions.")
    parser.add_argument("--query", default='', help="A summary of the nature of the transaction at issue in this document.")
    parser.add_argument("--top_k", type=int, default=50, help="Number of partitions to retrieve and rerank.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M",
                        help="Name of the generative model to use (for Qwen, use Qwen/Qwen2.5-7B-Instruct-1M).")
    parser.add_argument("--rerank", action='store_true', help="Enable reranking steps for both partitions and candidate answers.")
    args = parser.parse_args()
    
    # Load and prepare partitions
    partitions = read_partitions(args.file_path)
    logging.info("Loaded %d partitions from file.", len(partitions))
    partition_texts = [p['text'] for p in partitions]
    
    # Load embedding model and compute embeddings
    logging.info("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Computing embeddings for partitions...")
    embeddings = embedder.encode(partition_texts, convert_to_numpy=True)
    
    # Build FAISS index for retrieval
    index = build_faiss_index(embeddings)
    
    # Compute query embedding and retrieve relevant partitions
    logging.info("Computing embedding for the query...")
    query_embedding = embedder.encode(args.query, convert_to_numpy=True)
    logging.info("Retrieving top %d partitions relevant to the query...", args.top_k)
    retrieved_partitions = retrieve_context(query_embedding, index, partitions, args.top_k)
    
    # Load cross-encoder for reranking (if rerank flag is enabled)
    if args.rerank:
        logging.info("Loading cross-encoder for reranking...")
        try:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        except Exception as e:
            logging.error("Error loading cross-encoder: %s", e)
            sys.exit(1)
        logging.info("Reranking retrieved partitions based on query relevance...")
        top_partitions = rerank_partitions(args.query, retrieved_partitions, cross_encoder, args.top_k)
    else:
        logging.info("Skipping partition reranking. Using raw retrieved partitions.")
        top_partitions = retrieved_partitions
    
    # Log selected partition titles
    for idx, part in enumerate(top_partitions):
        section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
        logging.info("Selected Partition %d Section Name(s): %s", idx, section_names)
    
    # Concatenate texts from top partitions to form the refined context
    context = "\n\n".join([part['text'] for part in top_partitions])
    logging.info("Constructed refined context from top partitions.")
    
    # Optimized prompt: combine context with a clear task instruction
    optimized_prompt = (
        f"Below is context extracted from a document regarding an M&A transaction:\n\n{context}\n\n"
        "Please provide a concise and clear summary highlighting the key aspects of the transaction, including involved parties, "
        "basic deal terms, transaction structure, timeline, sales process details, board decisions, financial advisor roles, "
        "governance mechanisms, and any potential conflicts. Ensure the summary is well-structured and informative."
    )
    logging.info("Constructed optimized prompt for generation.")
    
    # Load tokenizer and initialize vllm generative model
    logging.info("Loading tokenizer for generative model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    logging.info("Loading vllm generative model: %s", args.model_name)
    llm = LLM(model=args.model_name,
              tensor_parallel_size=1,
              max_model_len=510000,
              enable_chunked_prefill=True,
              max_num_batched_tokens=131072,
              enforce_eager=True,
              quantization="fp8",  # Optionally enable FP8 quantization for reduced memory usage.
             )
    
    # Set sampling parameters for generation (these mirror Qwen's defaults)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    
    # Generate candidate answers and optionally rerank them using vllm and cross-encoder
    if args.rerank:
        logging.info("Generating and reranking candidate answers using vllm...")
        final_answer = generate_and_rerank(optimized_prompt, tokenizer, llm, sampling_params, cross_encoder, num_candidates=5)
    else:
        logging.info("Generating candidate answer without reranking...")
        final_answer = generate_without_rerank(optimized_prompt, tokenizer, llm, sampling_params)
    
    # Output the final answer
    print("=== Answer ===")
    print(final_answer)

if __name__ == "__main__":
    main()
