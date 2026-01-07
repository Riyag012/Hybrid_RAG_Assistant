# 🕸️ Hybrid GraphRAG Financial Assistant

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Riya012/Hybrid_GraphRAG_Assistant)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)](Dockerfile)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A state-of-the-art **Hybrid Retrieval-Augmented Generation (RAG)** engine designed to solve the "Disconnected Context" problem in complex documents. 

Whether it's **Financial Reports, Legal Contracts, Medical Records, or Technical Manuals**, this system combines the speed of **Vector Search** with the reasoning capabilities of **Knowledge Graphs** to answer questions that standard AI misses.


---

## 🎯 The Problem: Why Standard RAG Fails
Standard RAG (Retrieval Augmented Generation) relies solely on **Vector Similarity**. While good for general text, it fails catastrophically on financial documents (like 10-Ks or 10-Qs) because:
1.  **"The Needle in the Haystack":** Specific numbers (e.g., "$5.9 Billion") often look like "generic" text to a vector model, causing it to miss critical rows in dense tables.
2.  **Disconnected Context:** A revenue figure on Page 15 might be explained by a footnote on Page 30. Standard RAG treats these as separate chunks, losing the relationship.
3.  **Hallucination:** When standard RAG can't find the exact number, the LLM often guesses.

## 💡 The Solution: Hybrid GraphRAG
This project implements a **Hybrid Architecture** that establishes a "performance floor" using Vector Search and a "performance ceiling" using Graph connections.

### How it Works 
We utilize a **Hybrid Retrieval Strategy** with Neo4j's `OPTIONAL MATCH` logic:
* **Step 1 (Vector):** Find text chunks that are semantically similar to the question.
* **Step 2 (Graph):** *Enrich* those chunks with connected entities (e.g., `(Tesla)-[:HAS_CEO]->(Elon Musk)`).
* **Step 3 (Multimodal):** Use **Llama-Vision** to "see" and transcribe charts and tables that text parsers skip.

> **Result:** If the Graph has the answer, we get precision. If the Graph misses it (e.g., a standalone number), Vector Search catches it. We get the best of both worlds.

---

## ⚔️ Architecture Comparison: When to use What?

Not all RAG systems are created equal. Here is why we chose a **Hybrid** approach for financial data:

| Architecture | Best Used For... | Failure Mode |
| :--- | :--- | :--- |
| **Standard RAG**<br>*(Vector Only)* | **Simple Queries on Text.**<br>Good for summarization or finding broad concepts ("What is the company's mission?"). | **The "Needle in the Haystack".**<br>Fails to find specific numbers (e.g., "$10.5M") because embedding models treat numbers as generic tokens. |
| **GraphRAG**<br>*(Graph Only)* | **Complex Relationships.**<br>Good for multi-hop reasoning ("Who is the CEO of the subsidiary of X?"). | **Unstructured Data.**<br>If an entity (like a specific date or number) isn't extracted into the graph, the system returns nothing. |
| **Hybrid RAG**<br>*(Our Solution)* | **Financial, Legal Analysis etc.**<br>Combines the coverage of Vector Search with the precision of Knowledge Graphs. | **Computation Cost.**<br>Requires maintaining both a Vector Index and a Graph Database (solved here via Neo4j). |

> **Why Hybrid Wins:** We use `OPTIONAL MATCH` logic. If the Graph has the answer, we get precision. If the Graph misses it (e.g., a standalone number), Vector Search catches it. **We get the best of both worlds.**

## 📊 Benchmark Results (The Proof)
We benchmarked this system against a Standard RAG implementation using **30 "Hard" Financial Questions** from Tesla and Apple 2023 reports.

| Metric | GraphRAG (Hybrid) | Standard RAG | Result |
| :--- | :--- | :--- | :--- |
| **Wins** | **22** | 6 | 🏆 **GraphRAG Dominates** |
| **Ties** | 2 | 2 | - |
| **Accuracy** | **73.3%** | 20.0% | **+53% Improvement** |

**Visual Performance:**
* **Standard RAG:** Often replied "I don't know" or missed table rows.
* **GraphRAG:** Successfully retrieved exact dollar amounts, percentages, and cross-document relationships.

---

## 🏗️ Architecture & Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Python)
* **Database:** [Neo4j AuraDB](https://neo4j.com/cloud/aura/) (Graph + Vector Store)
* **LLM & Vision:** Groq (`Llama-3.3-70b` for text, `Llama-4-Scout` for vision)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Deployment:** Docker container on Hugging Face Spaces.

### 🖼️ Application Interface


![App UI](ss/Screenshot%202026-01-07%20150819.png)
![App UI](ss/Screenshot%202026-01-07%20151000.png)

---

## 🚀 How to Run

### Option 1: Live Demo
Try the deployed application on Hugging Face Spaces:
**[🔗 Click Here to Open App](https://huggingface.co/spaces/Riya012/Hybrid_GraphRAG_Assistant)**

### Option 2: Run with Docker (Recommended)
You can run the exact same container locally.

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/yourusername/Hybrid_RAG_Assistant.git](https://github.com/yourusername/Hybrid_RAG_Assistant.git)
    cd Hybrid_RAG_Assistant
    ```

2.  **Build the Image:**
    ```bash
    docker build -t graphrag-app .
    ```

3.  **Run the Container:**
    *(Make sure you have your API keys ready)*
    ```bash
    docker run -p 7860:7860 \
      -e GROQ_API_KEY="your_key" \
      -e NEO4J_URI="your_uri" \
      -e NEO4J_PASSWORD="your_password" \
      graphrag-app
    ```
    Access the app at `http://localhost:7860`.

### Option 3: Local Python
```bash
pip install -r requirements.txt
streamlit run app.py

```

---

## 📂 Project Structure

```text
├── app.py                 # Main application logic (Ingestion + Querying)
├── Dockerfile             # Container configuration for Cloud Deployment
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
└── ss/                # Screenshots and images

```
