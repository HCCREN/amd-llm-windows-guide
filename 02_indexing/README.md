In this step, we turn your PDF documents into a "searchable visual format" using the ColPali model.

# Why "Multi-Vector"?
Unlike traditional methods that squash a whole page into one number, **ColPali** creates ~1030 vectors for *each page*. This allows the AI to "see" charts, tables, and tiny details in your documents.

# Instructions

1. **Prepare Data**:
   Create a folder named `data` in the root directory (if it doesn't exist).
   Put your PDF files (e.g., invoices, papers, manuals) into the `data` folder.

2. **Run Ingestion**:
   Execute the script to start indexing:
   
   python 02_indexing/ingest_pdf.py

3. What happens next?

   a.The script will convert PDF pages to images.
   b.ColPali will analyze each image on your AMD GPU.
   c.The vectors will be stored in Qdrant (which you started in Step 0).

4. Verify: Once you see 🎉 Ingestion complete!, your data is ready to be searched.
   
   Note on Performance: Indexing is compute-intensive. On an RX 9070 XT, expect it to take about 1-2 seconds per page.











