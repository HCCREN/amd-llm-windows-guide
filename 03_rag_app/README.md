This is the final application! It combines retrieval (Step 2) with generation (Step 3).

# How it works (The RAG Pipeline)
1. **You ask a question**: "What is the profit in Q3?"
2. **Retriever (ColPali)**: Searches your Qdrant database for the most relevant PDF page using visual vectors.
3. **Smart Switching**: The app automatically unloads ColPali from your GPU and loads Qwen2-VL to save memory.
4. **Generator (Qwen2-VL)**: It looks at the *image* of that specific page and answers your question based on the visual data (charts, tables, text).

# Usage
Run the application:

python 03_rag_app/visual_rag_app.py

##Tips
Memory Management: If you see "Offloading...", don't worry! This is a feature to ensure the app runs smoothly on 16GB VRAM cards like the RX 9070 XT.
First Run: The first time you ask a question, it might take a few seconds to load the Qwen model. Subsequent questions will be faster if the model is already loaded.
