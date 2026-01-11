import os
import torch
import uuid
import gc
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColPali, ColPaliProcessor

# --- Configuration ---
DATA_DIR = "./data"
COLLECTION_NAME = "colpali_final_gpu"
QDRANT_URL = "http://localhost:6333"

# Auto-detect GPU (Targeting Device 1 for discrete GPU setups)
DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
if not torch.cuda.is_available():
    DEVICE = "cpu"
    print("⚠️ WARNING: No GPU detected. Ingestion will be very slow.")

client = QdrantClient(url=QDRANT_URL)

def setup_collection():
    """Initializes the Qdrant collection with Multi-Vector (MaxSim) support."""
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=128, 
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        )
        print(f"✨ Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"📚 Using existing collection: '{COLLECTION_NAME}'")

def is_already_indexed(filename):
    """Checks if a file is already inside the database to avoid duplicates."""
    res = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="filename", match=models.MatchValue(value=filename))]
        ),
        limit=1
    )
    return len(res[0]) > 0

def run_ingestion():
    # 1. Ensure DB is ready
    try:
        setup_collection()
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("   Did you run '00_infrastructure/start_qdrant.bat'?")
        return

    # 2. Load Model (ColPali)
    print(f"🚀 Loading ColPali on {DEVICE}...")
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2", 
        torch_dtype=torch.bfloat16, 
        device_map=DEVICE
    ).eval()
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    # 3. Find PDFs
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created '{DATA_DIR}' folder. Please put your PDF files there!")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDF files found in {DATA_DIR}. Please add some files.")
        return

    # 4. Process Loop
    for file in pdf_files:
        if is_already_indexed(file):
            print(f"⏭️ Skipping {file} (Already indexed).")
            continue

        print(f"📖 Processing: {file}...")
        path = os.path.join(DATA_DIR, file)
        
        try:
            # Convert PDF to Images (150 DPI is balanced for speed/quality)
            images = convert_from_path(path, dpi=150)
            
            for i, img in enumerate(images):
                # Process Image -> Multi-Vector
                inputs = processor.process_images([img]).to(DEVICE)
                
                with torch.no_grad():
                    embeddings = model(**inputs)
                    # Convert Tensor to List[List[float]]
                    vector_list = embeddings[0].cpu().float().numpy().tolist()
                
                # Generate unique ID
                point_id = str(uuid.uuid4())
                
                # Upload to Qdrant
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=vector_list,
                            payload={
                                "filename": file, 
                                "page": i + 1,
                                "type": "pdf_page"
                            }
                        )
                    ]
                )
                print(f"   ✅ Indexed Page {i+1} (ID: {point_id[:8]}...)")
                
                # Cleanup VRAM
                del embeddings, vector_list
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    print("\n🎉 Ingestion complete! You are ready for the RAG App.")

if __name__ == "__main__":
    run_ingestion()
