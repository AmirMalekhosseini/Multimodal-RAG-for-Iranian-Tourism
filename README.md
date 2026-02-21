# Multimodal-RAG-for-Iranian-Tourism

A reproducible Retrieval-Augmented Generation (RAG) system built for Iranian tourism. This project combines text and image retrieval over a curated dataset of ~1,500 locations across 30 provinces with an active, self-correcting generation loop to improve question-answering accuracy about places, attractions, and travel logistics in Iran.

Key ideas:
- Multimodal retrieval: retrieve both textual and visual evidence for each query.
- RAG-style generation: condition a generative model on retrieved passages and images.
- Active self-correction: generate, validate against retrieved evidence, and iteratively refine answers to reduce hallucinations.

---

Table of contents
- [Demo / Quick start](#demo--quick-start)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Model & architecture overview](#model--architecture-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Reproducibility notes](#reproducibility-notes)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License & citation](#license--citation)
- [Contact](#contact)

---

Demo / Quick start
1. Clone the repo:
   ```
   git clone https://github.com/AmirMalekhosseini/Multimodal-RAG-for-Iranian-Tourism.git
   cd Multimodal-RAG-for-Iranian-Tourism
   ```
2. Create a virtual environment and install dependencies (see Requirements below).
3. Launch the primary notebook:
   ```
   jupyter notebook notebooks/Multimodal-RAG.ipynb
   ```
4. Follow the notebook cells to:
   - preprocess the dataset,
   - build text and image embeddings,
   - create a vector index (FAISS),
   - run retrieval and generation, and
   - inspect the active self-correction loop and evaluation metrics.

Repository structure
- notebooks/
  - Multimodal-RAG.ipynb — main end-to-end notebook demonstrating preprocessing, indexing, retrieval, generation, and evaluation.
  - experiments/ — optional example experiments and ablation studies (if present).
- data/
  - raw/ — raw dataset files (images, CSV/JSON metadata).
  - processed/ — preprocessed artifacts and embeddings (not tracked in repo).
- src/ (optional) — helper scripts or modules extracted from notebooks for reuse.
- README.md — this file.

Dataset
- Custom dataset of ~1,500 locations covering 30 provinces in Iran.
- Each record typically contains:
  - location name
  - province
  - textual description (history, attractions, travel notes)
  - images (1..N photos)
  - geolocation and metadata (optional)
- Responsible use:
  - All images and text should be used respecting copyright and privacy.
  - If you include third-party images, ensure you have the right to redistribute or use them for model development.

Model & architecture overview
- Encoders:
  - Text encoder: sentence-transformers or another embedding model to compute semantic text embeddings.
  - Image encoder: CLIP (or another vision-language encoder) to compute image embeddings.
- Indexing / retrieval:
  - FAISS index for fast nearest-neighbor search.
  - Optionally separate indices for text and images, or a fused multimodal index.
- Generator:
  - A sequence-to-sequence or language model (e.g., a transformer) that conditions on retrieved text and image captions/representations to produce answers.
- Active self-correction:
  - After an initial generation, validate answer statements against retrieved evidence (e.g., check factual claims against retrieved passages).
  - If inconsistencies detected, run a refinement step that asks the generator to correct or re-answer using flagged evidence.

Requirements
- Python 3.8+
- Jupyter Notebook
- Typical Python packages used in the notebooks (example):
  - torch
  - transformers
  - sentence-transformers
  - faiss-cpu (or faiss-gpu)
  - datasets
  - huggingface-hub
  - openai (if using OpenAI APIs)
  - pillow
  - torchvision
  - tqdm
  - pandas, numpy
- GPU recommended for embedding/model inference but CPU mode works for small experiments.

Example install (create venv first)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter torch transformers sentence-transformers faiss-cpu datasets pillow torchvision tqdm pandas numpy
# install faiss-gpu instead of faiss-cpu if you have CUDA and prefer GPU indexing
```

How to run
- Open the main notebook and run cells in order. The notebook is organized into:
  1. Environment setup and imports
  2. Data loading and inspection
  3. Text and image preprocessing
  4. Embedding computation
  5. Index creation and evaluation of retrieval quality
  6. RAG generation pipeline
  7. Active self-correction loop and final evaluation
- Tips:
  - Precompute and cache embeddings to avoid long re-computation.
  - Use smaller subset of data when iterating on pipeline design.
  - If using an external LLM API (e.g., OpenAI), set your credentials in environment variables (e.g., OPENAI_API_KEY) or the method your notebook expects.

Reproducibility notes
- Keep seeds fixed where possible for deterministic embedding/model outputs.
- Save model and embedding versions (e.g., model names/tags, tokenizer versions).
- Cache intermediate artifacts (processed images, embeddings, FAISS indexes) and store checksums to verify unchanged inputs.

Evaluation
- Retrieval:
  - Recall@k, MRR (mean reciprocal rank) for text and image retrieval tasks.
- Generation:
  - Exact match / F1 for structured QA items (if you have ground-truth answers).
  - Human evaluation for fluency and factual correctness.
- Self-correction effect:
  - Compare generation quality metrics before and after the active refinement loop to quantify improvements and reduction in hallucination rates.

Examples of queries supported
- "What are the main attractions in Isfahan and how do I get there?"
- "Show me the historical highlights of Shiraz and any nearby UNESCO sites."
- "Which province has the best hiking trails for April?"

Privacy & ethics
- The system provides information about real-world places. Always cross-check critical travel information (safety, transport schedules, local regulations) with official sources before acting on it.
- Do not publish identifiable personal data extracted from images or text without consent.

Contributing
- If you'd like to contribute:
  1. Open an issue describing your idea or bug.
  2. Fork the repository and open a PR with clear description and tests (or updated notebooks).
  3. Prefer small, focused changes and include reproducible steps.

License & citation
- Add your preferred license file (e.g., MIT, Apache 2.0) to the repository.
- If you use this work in academic output, please cite the repository and include a short citation snippet here (update with authors, year, title, DOI if available).

Contact
- Author: Amir Malekhosseini
- GitHub: [AmirMalekhosseini](https://github.com/AmirMalekhosseini)

Acknowledgements
- Built using open-source models and libraries (Hugging Face Transformers, SentenceTransformers, FAISS, CLIP, PyTorch).
- Thanks to contributors and data curators for compiling the Iranian tourism dataset.

--- 
Notes
- This README is intended as a starting point. If you want, I can:
  - generate a requirements.txt from the notebooks,
  - extract and propose a lightweight CLI script to run indexing + RAG inference,
  - or add example Colab links and badges (GPU / CI / license).
