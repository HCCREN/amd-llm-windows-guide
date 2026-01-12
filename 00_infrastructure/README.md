# Step 0: Infrastructure Setup (Docker & Qdrant)

Before running any AI models on your AMD GPU, we need to set up the **Vector Database (Qdrant)**. 
We use Docker to run this database easily on Windows without needing complex installation.

## 1. Install Docker Desktop
1. Download **Docker Desktop for Windows** from the official website:
   [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Run the installer.
3. **Important**: If asked, ensure "Use WSL 2 based engine" is checked (this is standard for Windows 11).
4. After installation, restart your computer.
5. Launch "Docker Desktop" from your Start menu and wait until the bottom left corner says **"Engine Running"** (Green).

## 2. Start the Database
You have two options:

### Option A: The Easy Way (Double Click)
Simply double-click the `start_qdrant.bat` file in this folder. It will automatically check for Docker and start the database.

### Option B: The Manual Way (Command Line)
Open your terminal (Command Prompt or PowerShell) and run:

docker pull qdrant/qdrant
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant_amd_rag qdrant/qdrant

## 3. Verify it works
Open your web browser and visit:

http://localhost:6333/dashboard

If you see the Qdrant Dashboard, you represent "Infrastructure Ready"!
