#!/usr/bin/env python
"""
main.py

A robust retrieval-augmented generation (RAG) pipeline that:
1. Reads partitions from a JSON file.
2. Builds a FAISS index using embeddings.
3. Retrieves context based on a retrieval query.
4. Optionally reranks context based on the retrieval query if --rerank is enabled.
5. Generates multiple candidate summaries using Qwen via vllm (following its official prompting style) with a generation query.
6. Optionally uses a cross-encoder to select the best answer if --rerank is enabled.
7. Runs four fixed queries and saves each generated output in a txt file, including a section that lists the reranked partition section names and scores.

Usage:
    python main.py --file_path <path_to_json> [--top_k 100] [--model_name Qwen/Qwen2.5-7B-Instruct-1M] [--rerank]
    
The four queries are:
    A. "Golden parachute"
    B. "Conflicts of interest related to the transaction"
    C. "Litigation related to the transaction"
    D. "Background of the transaction"
"""
import os
import json
import argparse
import logging
import sys
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Set the XDG_CONFIG_HOME to a local directory (e.g., "./.config")
local_config = os.path.join(os.getcwd(), ".config")
os.makedirs(local_config, exist_ok=True)
os.environ["XDG_CONFIG_HOME"] = local_config

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
    """
    Rerank partitions using a cross-encoder.
    Returns:
        top_partitions: list of top partitions (up to top_k)
        ranked: full ranking list as tuples (partition, score)
    """
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
    return top_partitions, ranked

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
    
    # Replicate the prompt to generate multiple answers
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
    
    parser = argparse.ArgumentParser(
        description="Retrieval-Augmented Generation Pipeline with vllm-based Qwen Inference for multiple queries"
    )
    parser.add_argument("--file_path", default='data/example_sec.json', help="Path to the JSON file containing partitions.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of partitions to retrieve and rerank.")
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
    
    # If reranking is enabled, load cross-encoder (used for both partition and candidate reranking)
    cross_encoder = None
    if args.rerank:
        logging.info("Loading cross-encoder for reranking...")
        try:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        except Exception as e:
            logging.error("Error loading cross-encoder: %s", e)
            sys.exit(1)
    
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
    
    # Define the fixed queries to run, along with a label for the output file.
    queries = [
        ("A", "Golden parachute"),
        ("B", "Conflicts of interest related to the transaction"),
        ("C", "Litigation related to the transaction"),
        ("D", "Background of the transaction"),
    ]
    
    # Loop through each query, run the retrieval-augmented generation pipeline, and save the output.
    for label, query_text in queries:
        logging.info("Processing query %s: %s", label, query_text)
        # Compute embedding for the current query (using the query text for retrieval)
        retrieval_query_embedding = embedder.encode(query_text, convert_to_numpy=True)
        logging.info("Retrieving top %d partitions relevant to the query...", args.top_k)
        retrieved_partitions = retrieve_context(retrieval_query_embedding, index, partitions, args.top_k)
    
        # Rerank partitions if enabled; otherwise, use the raw retrieved partitions.
        reranked_details = None
        if args.rerank:
            logging.info("Reranking retrieved partitions for query: %s", query_text)
            top_partitions, reranked_details = rerank_partitions(query_text, retrieved_partitions, cross_encoder, args.top_k)
        else:
            logging.info("Skipping partition reranking for query: %s. Using raw retrieved partitions.", query_text)
            top_partitions = retrieved_partitions
    
        # Log selected partition titles
        for idx, part in enumerate(top_partitions):
            section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
            logging.info("Selected Partition %d Section Name(s): %s", idx, section_names)
    
        # Concatenate texts from top partitions to form the refined context
        context = "\n\n".join([part['text'] for part in top_partitions])
        logging.info("Constructed refined context from top partitions.")
    
        # Construct an optimized prompt using the current query.
        optimized_prompt = (
            f"Below is context extracted from a document regarding {query_text}:\n\n{context}\n\n"
            f"Please extract important information about {query_text}."
        )
    
        logging.info("Constructed optimized prompt for query %s.", label)
    
        # Generate candidate answers and optionally rerank them using vllm and cross-encoder
        if args.rerank:
            logging.info("Generating and reranking candidate answers for query %s using vllm...", label)
            final_answer = generate_and_rerank(optimized_prompt, tokenizer, llm, sampling_params, cross_encoder, num_candidates=5)
        else:
            logging.info("Generating candidate answer for query %s without reranking...", label)
            final_answer = generate_without_rerank(optimized_prompt, tokenizer, llm, sampling_params)
    
        # Prepare the output text.
        output_text = f"=== Answer for Query {label} ({query_text}) ===\n\n{final_answer}\n"
    
        # If reranking is enabled, add a separate section with the reranked partition details.
        if reranked_details is not None:
            output_text += "\n=== Reranked Partitions and Scores ===\n"
            for idx, (part, score) in enumerate(reranked_details):
                section_names = ", ".join(part['titles']) if part['titles'] else "No Title"
                output_text += f"Rank {idx+1}: {section_names} | Score: {score:.4f}\n"
    
        # Save the final answer (and reranking details) to a text file named output_<label>.txt
        output_filename = f"output_{label}.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as outfile:
                outfile.write(output_text)
            logging.info("Saved generated output for query %s to %s", label, output_filename)
        except Exception as e:
            logging.error("Error writing output for query %s: %s", label, e)
    
        # Optionally, also print the answer to stdout
        print(output_text)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
