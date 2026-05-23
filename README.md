# 🏏 CricRAG - Cricket Intelligence Workspace

CricRAG is an offline-first cricket assistant that combines local semantic search (ChromaDB + sentence-transformers), structured player profiles, and MCP-style routing reasoning loops. It features a premium stadium-themed web dashboard and an embedded conversational assistant.

This guide contains instructions to run CricRAG locally and deploy it quickly to cloud platforms.

---

## 🔒 Security Warning
Before pushing this code to a public repository (like GitHub):
* Ensure **`settings.json`** is ignored. A `.gitignore` file has been prepared to protect your credentials.
* **Never** commit your Gemini API Key or other secrets.

---

## 🚀 Quick Deployment Guide

To deploy this project to the cloud, you should first initialize a Git repository and push it to GitHub.

### Step 1: Push your code to GitHub
Run the following commands in your project folder:
```bash
git init
git add .
git commit -m "Initial commit with deployment readiness configuration"
git branch -M main
# Add your remote repository and push:
# git remote add origin https://github.com/yourusername/cricrag.git
# git push -u origin main
```

---

### Option A: Deploy to Hugging Face Spaces (Easiest & Free)
Hugging Face Spaces supports running Docker apps or Python/Gradio apps directly:

#### Method 1: Using Docker (Recommended for exact setup)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set a **Space name** and choose **Docker** as the SDK.
3. Select **Blank** as the Docker template.
4. Set the Space to **Public** or **Private**.
5. Once created, click on **Settings** in the Space menu, find **Variables and Secrets**, and add:
   * **Secret**: `GEMINI_API_KEY` = `your-gemini-api-key`
6. Push this repository to the Hugging Face Git remote, or connect your GitHub repository to build automatically.

#### Method 2: Gradio Space (Alternative)
1. Set the SDK to **Gradio**.
2. Hugging Face expects the entrypoint to be `app.py`, which is already prepared.
3. The Space will automatically read `requirements.txt` and run the app.
4. Add your `GEMINI_API_KEY` under Space Secrets.

---

### Option B: Deploy to Render (Web Service)
Render is a cloud hosting provider that makes Docker deployments easy:

1. Sign in to [Render](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository containing the CricRAG code.
4. Configure the Web Service:
   * **Runtime**: `Docker`
   * **Instance Type**: Free (or higher)
5. Click **Advanced** and add the following Environment Variable:
   * `GEMINI_API_KEY` = `your-gemini-api-key`
6. Click **Deploy Web Service**. Render will build the Docker container and expose the application dynamically.

---

### Option C: Deploy to Railway (Fastest container deployment)
1. Sign in to [Railway](https://railway.app).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select your repository.
4. Click **Variables** in your service panel and add:
   * `GEMINI_API_KEY` = `your-gemini-api-key`
5. Click **Deploy**. Railway will automatically detect the `Dockerfile` and boot the container.

---

## 🛠️ Environment Configurations

CricRAG natively reads configurations from system environment variables during cloud deployments to bypass writing to disk:

| Variable Name | Description | Default Value |
| --- | --- | --- |
| `PORT` | Dynamically assigned server port (automatically managed by PaaS hosts). | `7861` |
| `GEMINI_API_KEY` | Your Google Gemini API Key. Setting this automatically sets Gemini as the default LLM provider. | Empty |
| `OLLAMA_ENDPOINT` | The URL of a local/hosted Ollama instance. | `http://localhost:11434` |

---

## 🏃 Local Execution

If you want to run the project locally, install dependencies and launch the server:
```bash
pip install -r requirements.txt
python app.py
```
Open [http://localhost:7861](http://localhost:7861) in your browser.
