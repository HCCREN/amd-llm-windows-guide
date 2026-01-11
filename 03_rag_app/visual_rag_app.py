import os
import torch
import gc
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path

# --- Configuration for AMD RDNA 4 ---
# Essential for preventing fragmentation on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_DIR = "./data"
COLLECTION_NAME = "colpali_final_gpu"
QDRANT_URL = "http://localhost:6333"

# Smart Device Selection
DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
print(f"🚀 Initializing AMD Visual RAG on {torch.cuda.get_device_name(DEVICE)}...")

# --- Global Variables for Offloading ---
client = QdrantClient(url=QDRANT_URL)
retriever_model = None
retriever_processor = None
generator_model = None
generator_processor = None

def clear_vram():
    """Aggressively clears VRAM to prevent OOM on Windows"""
    gc.collect()
    torch.cuda.empty_cache()

# --- Component 1: Multi-Vector Retriever (The "Search Engine") ---
def load_retriever():
    global retriever_model, retriever_processor, generator_model
    
    # 1. Unload Qwen (Generator) if active
    if generator_model is not None:
        print("🔄 Offloading Generator (Qwen) to CPU...")
        generator_model.to("cpu")
        clear_vram()

    # 2. Load ColPali (Retriever)
    if retriever_model is None:
        print("📥 Loading Retriever (ColPali v1.2)...")
        # Use bfloat16 for RDNA 4 speedup
        retriever_model = ColPali.from_pretrained(
            "vidore/colpali-v1.2", 
            torch_dtype=torch.bfloat16, 
            device_map=DEVICE
        ).eval()
        retriever_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    else:
        print("⚡ Moving Retriever to GPU...")
        retriever_model.to(DEVICE)

# --- Component 2: Visual Generator (The "Brain") ---
def load_generator():
    global retriever_model, generator_model, generator_processor
    
    # 1. Unload ColPali (Retriever) if active
    if retriever_model is not None:
        print("🔄 Offloading Retriever (ColPali) to CPU...")
        retriever_model.to("cpu")
        clear_vram()

    # 2. Load Qwen2-VL
    if generator_model is None:
        print("📥 Loading Generator (Qwen2-VL)...")
        generator_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            _attn_implementation="eager" # Crucial for Windows Stability
        ).eval()
        generator_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    else:
        print("⚡ Moving Generator to GPU...")
        generator_model.to(DEVICE)

# --- The Main Pipeline ---
def run_rag_pipeline(user_query):
    # Step 1: Multi-Vector Retrieval
    load_retriever()
    print(f"🔍 Searching your personal data for: '{user_query}'")
    
    with torch.no_grad():
        # Convert text query to Multi-Vector embedding
        inputs = retriever_processor.process_queries([user_query]).to(DEVICE)
        embeddings = retriever_model(**inputs)
        query_vector = embeddings[0].cpu().float().numpy().tolist()

    # Query Qdrant (MaxSim calculation happens here inside Qdrant)
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1 # Retrieve the single best matching page
    ).points

    if not search_result:
        return "❌ I couldn't find any relevant data in your PDFs."

    # Parse Result
    top_hit = search_result[0]
    filename = top_hit.payload['filename']
    page_num = top_hit.payload['page']
    score = top_hit.score
    print(f"✅ Found relevant data: {filename} (Page {page_num}) | Score: {score:.2f}")

    # Step 2: Visual Generation
    load_generator()
    print("🤖 Analyzing the retrieved page...")

    # Load the specific page image
    pdf_path = os.path.join(DATA_DIR, filename)
    try:
        page_image = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)[0]
    except Exception as e:
        return f"❌ Error loading PDF page: {e}"

    # Prepare prompt for Qwen
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": page_image},
                {"type": "text", "text": f"Context: This is page {page_num} of {filename} from the user's personal data. Question: {user_query}. Answer strictly based on the image provided."},
            ],
        }
    ]

    # Generate Answer
    text = generator_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = generator_processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = generator_model.generate(**inputs, max_new_tokens=300)
        output_text = generator_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    # Extract clean answer
    if "assistant\n" in output_text:
        return output_text.split("assistant\n")[-1]
    return output_text

# --- Interactive Loop ---
if __name__ == "__main__":
    print("\n💡 System ready. Indexing is handled by 'ingest_pdf.py'.")
    print("   This script only searches existing data.")
    
    while True:
        q = input("\nVisual RAG > Ask about your data (q to quit): ")
        if q.lower() == 'q': break
        try:
            answer = run_rag_pipeline(q)
            print(f"\n💬 Answer:\n{answer}\n{'='*50}")
        except Exception as e:
            print(f"❌ Pipeline Error: {e}")
            clear_vram()
