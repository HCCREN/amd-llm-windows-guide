This step ensures your Python environment is ready to talk to your AMD GPU (RX 9070 XT).

# 1. Create a Virtual Environment (Recommended)
Open your terminal in the root folder and run:

python -m venv venv
.\venv\Scripts\activate

# 2. Install PyTorch for AMD (ROCm)
⚠️ CRITICAL STEP: Do not just run pip install torch. You must point to the AMD ROCm repository.
Run this exact command (updated for ROCm 6.2/7.x preview):

pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/rocm6.2](https://download.pytorch.org/whl/nightly/rocm6.2)

Note: If the above link fails, check the official PyTorch Get Started page for the latest "Preview (Nightly)" link for Windows.

# 3. Install Other Dependencies
Once PyTorch is installed, install the rest of the libraries:

pip install -r 01_setup/requirements.txt

# 4. Install Poppler (Required for PDF processing)
The pdf2image library requires Poppler.
Download the latest Release from Poppler for Windows.
Extract the zip file.
Add the bin folder path (e.g., C:\Program Files\poppler-xx\bin) to your System Environment Variables PATH.

# 5. Verify Installation

python 01_setup/verify_gpu.py




















