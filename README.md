# SWEET-Externship-Project
LOCAL DOCUMENT Q&A RAG PIPELINE

A local Retrieval-Augmented Generation (RAG) system built with Python, NumPy, and Ollama. This application indexes plain-text documents into vector embeddings and provides an interactive CLI for a Q&A with direct source citations.

Technical Architecture and Pipeline

Document Chunking: Reads the target text file and splits content on double-newline (\n\n) delimiters to form paragraph-based chunks.

Vector Embedding & Normalization: Converts text chunks into vector embeddings using Ollama's nomic-embed-text model. Each vector undergoes L2 normalization upon generation.

Similarity Search: Computes query vectors using the same model and performs fast cosine similarity scoring across all indexed chunks using NumPy dot products. The top K (K=4) scoring chunks are retrieved.  

Contextual Generation: Sends the retrieved context and user prompt to llama3.2:3b following instructions to guarantee accurate answers and eliminate hallucinations.

Set up:

Create a virtual environment to install packages locally.

Install the packages numpy and ollama via the requirments.txt.

Install Ollama to pull nomic-embed-text and llama3.2:3b as local AI models.

Run main.py and follow the instructions.

Design Decisions:

L2 Normalization at Indexing: Normalizing all vectors during creation reduces cosine similarity calculations to a simple dot product, avoiding repeated calculations.

Paragraph-Based Chunking: Splits text by natural paragraphs instead of fixed line counts to keep complete sentences together, preserve semantic context, and prevent key information from being cut mid-thought

Model Selection: Selected a 3B parameter model to minimize response latency during interactive testing while retaining sufficient reasoning capacity to strictly follow prompt guardrails and prevent hallucinations.



