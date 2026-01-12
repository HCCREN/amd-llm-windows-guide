# 🚀 AMD LLM Windows Guide: Visual RAG Edition (RX 9070 XT / RDNA 3+)

> **A complete, step-by-step guide to running advanced Multi-Modal RAG on AMD Radeon GPUs in a pure Windows environment.** > ❌ No Linux required  
> ❌ No WSL required  
> ✅ 100% Native Windows Python + ROCm

![AMD Radeon](https://img.shields.io/badge/GPU-AMD_Radeon_RX_9070_XT-red) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![ROCm](https://img.shields.io/badge/ROCm-6.2%2B-orange)

## 🎯 Project Goal
This project demonstrates how to build a **Visual Retrieval-Augmented Generation (RAG)** system using your local AMD GPU.
Instead of standard text search, we use **ColPali (Multi-Vector Retrieval)** to "see" documents and **Qwen2-VL** to answer questions based on charts, tables, and layouts.

## 🛠️ Tech Stack
* **Hardware**: AMD Radeon RX 9070 XT (or any RDNA 3/4 GPU)
* **OS**: Windows 10/11 (Native)
* **Database**: Qdrant (via Docker)
* **Models**: 
    * Retriever: `vidore/colpali-v1.2` (Visual Embeddings)
    * Generator: `Qwen/Qwen2-VL-2B-Instruct` (Visual LLM)

---

## 📚 Step-by-Step Guide

Follow these folders in order to build your own app:

### [Step 0: Infrastructure Setup](./infrastructure)
**Start here!** Set up the Vector Database (Qdrant) using Docker. 
* *Includes a one-click `.bat` launcher.*

### [Step 1: Python Environment & ROCm](./01_setup)
Install PyTorch for AMD (ROCm) and verify your GPU is detected correctly.
* *Solves the "PyTorch not finding GPU" issue on Windows.*

### [Step 2: Indexing Your Data](./02_indexing)
Learn how to convert your PDF documents into multi-vector visual embeddings.
* *Uses ColPali to read charts and tables.*

### [Step 3: Run the Visual RAG App](./03_rag_app)
**The Final Product!** run a chat application that switches models automatically to fit in 16GB VRAM.
* *Features automatic Model Offloading (CPU <-> GPU).*

---

## ⚡ Quick Start (For Experienced Users)

1.  **Start Database**: Run `infrastructure/start_qdrant.bat`
2.  **Install Deps**: 
    ```bash
    pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/rocm6.2](https://download.pytorch.org/whl/nightly/rocm6.2)
    pip install -r 01_setup/requirements.txt
    ```
3.  **Add Data**: Put PDFs in `data/` folder.
4.  **Index**: `python 02_indexing/ingest_pdf.py`
5.  **Run**: `python 03_rag_app/visual_rag_app.py`

---

## 🤝 Contribution
If you find this guide helpful, please give it a ⭐ **Star**!
Issues and Pull Requests are welcome to help more AMD users on Windows.
