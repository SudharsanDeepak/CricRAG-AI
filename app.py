import os
import json
import gradio as gr
from rag_engine import RAGEngine
import mcp_agent
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize RAG Engine on startup asynchronously in a background thread
# to prevent blocking uvicorn startup or page loading.
rag_engine = None
is_engine_loading = True

def init_rag_async():
    global rag_engine, is_engine_loading
    try:
        print("Initializing CricRAG Engine in background thread...")
        # Force a small sleep to let uvicorn finish starting up
        time.sleep(1.0)
        from rag_engine import RAGEngine
        rag_engine = RAGEngine()
        
        # Auto-ingest knowledge base if empty
        stats = rag_engine.get_db_stats()
        if stats["total_chunks"] == 0:
            print("Knowledge base is empty. Pre-loading 1000+ cricket facts in background...")
            rag_engine.ingest_directory(os.path.join(BASE_DIR, "knowledge_base"))
            stats = rag_engine.get_db_stats()
            print(f"Pre-loaded successfully! Total chunks: {stats['total_chunks']}")
        else:
            print(f"Database loaded successfully! Total chunks: {stats['total_chunks']}")
    except Exception as e:
        print(f"Engine initialization deferred or failed: {e}")
        rag_engine = None
    finally:
        is_engine_loading = False

threading.Thread(target=init_rag_async, daemon=True).start()

# Settings persistence
if os.environ.get("SPACE_ID"):
    SETTINGS_FILE = "/tmp/settings.json"
else:
    SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
DEFAULT_SETTINGS = {
    "provider": "Google Gemini API" if os.environ.get("GEMINI_API_KEY") else "Offline Simulator (Pre-compiled & Heuristics)",
    "ollama_endpoint": os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except:
            pass
    return DEFAULT_SETTINGS

def save_settings(provider, ollama_endpoint):
    settings = {
        "provider": provider,
        "ollama_endpoint": ollama_endpoint
    }
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        return "Settings saved successfully!"
    except Exception as e:
        return f"Error saving settings: {e}"

# UI Functions
def chat_response(message, history, mode, settings_state):
    if not message.strip():
        return "", history, ""
        
    global rag_engine, is_engine_loading
    
    # Wait up to 5 seconds if the engine is currently loading in the background
    if rag_engine is None and is_engine_loading:
        print("Engine is still loading. Waiting for it...")
        for _ in range(10):
            if rag_engine is not None:
                break
            time.sleep(0.5)
            
    if rag_engine is None:
        if is_engine_loading:
            if history is None:
                history = []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "🏏 CricRAG Engine is still initializing (loading vector models and indices). Please wait a few seconds and try again!"})
            return "", history, "Engine is still loading in the background. Please wait."
        else:
            try:
                from rag_engine import RAGEngine
                rag_engine = RAGEngine()
                rag_engine.ingest_directory(os.path.join(BASE_DIR, "knowledge_base"))
            except Exception as err:
                if history is None:
                    history = []
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"System Error: RAG Engine failed to initialize ({err})"})
                return "", history, "Failed to initialize vector database. Make sure requirements are fully installed."
            
    provider = settings_state.get("provider", "Offline Simulator (Pre-compiled & Heuristics)")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    endpoint = settings_state.get("ollama_endpoint", "http://localhost:11434")

    gemini_model_name = ""

    thought_log = ""
    
    if mode == "Standard RAG":
        # Run standard context search and generate output
        # Perform semantic query
        results = rag_engine.query(message, n_results=4)
        context = " ".join([r["content"] for r in results]) if results else ""
        sources = ", ".join(list(set([r["source"] for r in results]))) if results else ""

        # Formulate response
        if provider == "Google Gemini API" and api_key.strip():
            try:
                if context:
                    prompt = (
                        f"You are an expert IPL assistant. Use the provided context when relevant, "
                        f"but you may also use your general cricket knowledge to answer accurately.\n\n"
                        f"Context:\n{context}\n\nQuestion: {message}\n\n"
                        f"Answer clearly and directly. If the context is incomplete, supplement it with general cricket knowledge, "
                        f"and mention when you are doing so."
                    )
                else:
                    prompt = (
                        f"You are an expert IPL and cricket assistant. Answer this cricket-related question using your general knowledge. "
                        f"If you are unsure about a specific detail, say so instead of inventing it.\n\nQuestion: {message}"
                    )
                ans, gemini_model_name = mcp_agent.generate_gemini_text(api_key, prompt)
            except Exception as e:
                if context:
                    ans = f"Error calling Gemini: {e}\n\nFallback Answer based on context:\n{context}\n\n(Sources: {sources})"
                else:
                    ans = f"Error calling Gemini: {e}\n\nPlease try again with a cricket-specific question."
        elif provider == "Local Ollama" and context:
            try:
                import requests
                url = f"{endpoint}/api/generate"
                prompt = f"Context: {context}\n\nQuestion: {message}\n\nAnswer based on context. Keep it grounded."
                payload = {"model": "llama3", "prompt": prompt, "stream": False}
                r = requests.post(url, json=payload, timeout=10)
                if r.status_code == 200:
                    ans = r.json().get("response", "").strip()
                else:
                    raise Exception(f"HTTP Status {r.status_code}")
            except Exception as e:
                ans = f"Error calling Ollama: {e}\n\nFallback Answer based on context:\n{context}\n\n(Sources: {sources})"
        else:
            if results:
                # Fallback Answer
                ans = f"**AI Cricket Assistant Answer (Offline RAG)**:\n\nBased on the retrieved IPL facts:\n- {results[0]['content']}\n- {results[1]['content'] if len(results) > 1 else ''}\n\n*(Source files: {sources})*"
            else:
                ans = "No matching facts found in the IPL database. Try asking a cricket-specific question or upload relevant files."
            
        thought_log = f"=== STANDARD RAG PIPELINE ===\n1. Received user question: '{message}'\n2. Running sentence-transformers query on ChromaDB...\n3. Found {len(results)} matching chunks.\n4. Formulating response via {provider}{f' ({gemini_model_name})' if gemini_model_name else ''}..."
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ans})
        
    else:
        # MCP Agent Mode (Tool calling)
        # Execute the agent loop
        if provider == "Google Gemini API" and api_key.strip():
            ans, steps = mcp_agent.run_llm_mcp_agent(message, rag_engine, "Gemini", api_key, endpoint)
        elif provider == "Local Ollama":
            ans, steps = mcp_agent.run_llm_mcp_agent(message, rag_engine, "Ollama", api_key, endpoint)
        else:
            # Offline simulator
            ans, steps = mcp_agent.run_offline_fallback_agent(message, rag_engine)
            
        # Format the thought log for the side panel
        log_str = "=== MCP AGENT TRACE ===\n"
        for i, step in enumerate(steps, 1):
            log_str += f"\n[STEP {i}] — Thinking Process:\n"
            log_str += f"💭 {step['thought']}\n"
            log_str += f"🛠️ Calling Tool: {step['tool_call']}\n"
            log_str += f"📥 Tool Output: {step['tool_output'][:300]}...\n"
            log_str += "-" * 40 + "\n"
        log_str += "\n🎯 Goal Achieved! Generating grounded final answer."
        thought_log = log_str
        
        # Append user message and agent's final answer to history
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ans})
        
    return "", history, thought_log

def get_stats_ui():
    global rag_engine
    if rag_engine is None:
        return "Database not initialized. Please click 'Test/Reload Engine'."
    try:
        s = rag_engine.get_db_stats()
        stats_str = f"### Database Statistics\n"
        stats_str += f"- **Total Text Chunks**: {s['total_chunks']}\n"
        stats_str += f"- **Unique Ingested Files**: {s['source_count']}\n"
        if s['unique_sources']:
            stats_str += "\n**Indexed Source Files:**\n"
            for src in s['unique_sources']:
                stats_str += f"- `{src}`\n"
        return stats_str
    except Exception as e:
        return f"Error loading database stats: {e}"

def upload_files(file_objs):
    if not file_objs:
        return "No files uploaded.", get_stats_ui()
        
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = RAGEngine()
        except Exception as e:
            return f"Error: Failed to load RAG engine: {e}", ""
            
    success_count = 0
    total_chunks_added = 0
    for file_obj in file_objs:
        try:
            # Gradio files are temp files, copy or ingest directly
            chunks = rag_engine.ingest_single_file(file_obj.name)
            if chunks > 0:
                success_count += 1
                total_chunks_added += chunks
        except Exception as e:
            print(f"Error ingesting uploaded file {file_obj.name}: {e}")
            
    return f"Indexed {success_count} file(s) successfully, adding {total_chunks_added} chunks to the vector database.", get_stats_ui()

def clear_db_ui():
    global rag_engine
    if rag_engine:
        rag_engine.clear_database()
        return "Vector database cleared!", get_stats_ui()
    return "Database not initialized.", ""

def search_explorer_ui(query_text):
    global rag_engine
    if not query_text.strip():
        return []
    if rag_engine is None:
        try:
            rag_engine = RAGEngine()
        except:
            return [["Error", "Database not loaded", 0.0, 0]]
            
    results = rag_engine.query(query_text, n_results=5)
    table_data = []
    for r in results:
        table_data.append([
            r["source"],
            r["content"],
            f"{r['score']*100:.2f}%",
            r["chunk_index"]
        ])
    return table_data

# Custom Premium CSS for styling the UI
CUSTOM_CSS = """
:root, .dark, .light {
    /* Gradio System Variables Overrides */
    --background-fill-primary: #070913 !important;
    --background-fill-secondary: #0F132A !important;
    --block-background-fill: #0F132A !important;
    --block-border-color: rgba(255, 255, 255, 0.08) !important;
    --border-color-primary: rgba(255, 255, 255, 0.08) !important;
    --border-color-secondary: rgba(255, 255, 255, 0.04) !important;
    --body-text-color: #FFFFFF !important;
    --body-text-color-subdued: #E2E8F0 !important;
    --block-title-text-color: #FFFFFF !important;
    --block-label-text-color: #FFFFFF !important;
    --input-background-fill: #070913 !important;
    --input-border-color: rgba(255, 255, 255, 0.08) !important;
    --input-text-color: #FFFFFF !important;
    --input-placeholder-color: #E2E8F0 !important;
    --button-primary-background-fill: #00FF87 !important;
    --button-primary-text-color: #070913 !important;
    --button-primary-background-fill-hover: #059669 !important;
    --button-secondary-background-fill: #0F132A !important;
    --button-secondary-text-color: #FFFFFF !important;
    --button-secondary-background-fill-hover: #070913 !important;
    --neutral-50: #070913 !important;
    --neutral-100: #0F132A !important;
    --neutral-200: rgba(255, 255, 255, 0.08) !important;
    --neutral-300: rgba(255, 255, 255, 0.12) !important;
    --neutral-400: #E2E8F0 !important;
    --neutral-500: #E2E8F0 !important;
    --neutral-600: #FFFFFF !important;
    --neutral-700: #FFFFFF !important;
    --neutral-800: #FFFFFF !important;
    --neutral-900: #FFFFFF !important;
    --neutral-950: #FFFFFF !important;

    /* Custom Theme Variables */
    --bg: #070913;
    --bg-2: #0F132A;
    --panel: rgba(15, 19, 42, 0.78);
    --panel-strong: #0F132A;
    --line: rgba(255, 255, 255, 0.08);
    --line-strong: rgba(255, 255, 255, 0.12);
    --text: #FFFFFF;
    --muted: #E2E8F0;
    --accent: #35A8E0;
    --accent-2: #4AE2B6;
    --accent-3: #F4D068;
    --danger: #FE5F55;
    --glow: rgba(53, 168, 224, 0.12);
}

* {
    box-sizing: border-box;
}

body {
    background: #070913 !important;
    color: #FFFFFF !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    background: transparent !important;
    color: #FFFFFF !important;
}

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 255, 135, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 135, 0.03) 1px, transparent 1px);
    background-size: 72px 72px, 72px 72px;
    mask-image: radial-gradient(circle at center, black 30%, transparent 100%);
    pointer-events: none;
    opacity: 0.35;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.4);
}

::-webkit-scrollbar-thumb {
    background: #35A8E0 !important;
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: #2086C2 !important;
}

.app-shell {
    max-width: 1360px;
    margin: 0 auto;
    padding: 28px 22px 42px;
}

.hero-band {
    background: #0F132A !important;
    border: 1px solid var(--line) !important;
    border-radius: 28px !important;
    padding: 24px 26px !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25) !important;
    margin-bottom: 18px !important;
    position: relative !important;
    overflow: hidden !important;
}

.hero-band::after {
    content: '';
    position: absolute;
    inset: 0;
    background: transparent;
    pointer-events: none;
}

.eyebrow {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    border: 1px solid rgba(0, 255, 135, 0.2) !important;
    background: rgba(0, 255, 135, 0.08) !important;
    color: var(--accent-green) !important;
    border-radius: 999px !important;
    padding: 7px 12px !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    position: relative !important;
    z-index: 1 !important;
}

.hero-title {
    margin-top: 14px !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
    font-size: clamp(2rem, 4.4vw, 4.6rem) !important;
    line-height: 0.98 !important;
    letter-spacing: -0.05em !important;
    font-weight: 900 !important;
    max-width: 11ch !important;
    position: relative !important;
    z-index: 1 !important;
    color: #FFFFFF !important;
}

.hero-title span {
    color: #35A8E0 !important;
}

.hero-copy {
    margin-top: 12px !important;
    max-width: 72ch !important;
    color: #5A7184 !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    position: relative !important;
    z-index: 1 !important;
}

.hero-meta {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    margin-top: 18px !important;
    position: relative !important;
    z-index: 1 !important;
}

.meta-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 9px 12px !important;
    border-radius: 999px !important;
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid var(--line) !important;
    color: #5A7184 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

.meta-pill strong {
    color: #FFFFFF !important;
}

.top-stats {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 14px !important;
    margin-bottom: 18px !important;
}

.stat-tile {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 22px !important;
    padding: 18px 18px 16px !important;
    box-shadow: 0 12px 30px rgba(27, 46, 60, 0.04) !important;
    backdrop-filter: blur(18px) !important;
}

.stat-kicker {
    color: #5A7184 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stat-number {
    margin-top: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    color: #FFFFFF !important;
}

.stat-text {
    margin-top: 6px !important;
    color: #5A7184 !important;
    font-size: 0.9rem !important;
    line-height: 1.55 !important;
}

.assistant-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.42fr) minmax(320px, 0.78fr);
    gap: 18px;
    margin-bottom: 18px;
}

.assistant-stage,
.insight-rail {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 26px;
    box-shadow: 0 12px 30px rgba(27, 46, 60, 0.04);
    backdrop-filter: blur(18px);
    overflow: hidden;
}

.assistant-stage {
    padding: 18px;
}

.insight-rail {
    padding: 18px;
    display: grid;
    gap: 14px;
    align-content: start;
}

.stage-banner,
.rail-card {
    border-radius: 22px;
    border: 1px solid var(--line);
    background: #ffffff;
    padding: 18px;
}

.stage-banner {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 14px;
    background: #ffffff;
}

.stage-label,
.rail-label {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    border: 1px solid rgba(53, 168, 224, 0.25);
    background: rgba(53, 168, 224, 0.08);
    color: #2086C2;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stage-banner h2,
.rail-card h3 {
    margin-top: 10px;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.04em;
    color: var(--text) !important;
}

.stage-banner h2 {
    font-size: clamp(1.4rem, 2vw, 2rem);
}

.stage-banner p,
.rail-card p,
.rail-card li {
    margin-top: 10px;
    color: var(--muted) !important;
    line-height: 1.65;
    font-size: 0.94rem;
}

.stage-badges {
    display: grid;
    gap: 8px;
    min-width: 188px;
}

.stage-badge,
.rail-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.9);
    color: var(--text) !important;
    font-weight: 700;
    font-size: 0.84rem;
}

.prompt-strip {
    margin-bottom: 14px;
    gap: 10px !important;
    flex-wrap: wrap !important;
}

.prompt-strip button {
    flex: 1 1 180px;
    min-height: 44px;
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    font-size: 0.86rem !important;
    font-weight: 700 !important;
}

.prompt-strip button:hover {
    transform: translateY(-1px) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 8px 18px rgba(27, 46, 60, 0.06) !important;
}

.main-card,
.settings-box,
.info-card,
.stat-card,
.gradio-chatbot,
#chat-header-row,
#input-container-row {
    background: var(--panel);
    border: 1px solid var(--line) !important;
    box-shadow: 0 12px 30px rgba(27, 46, 60, 0.04) !important;
    backdrop-filter: blur(18px);
}

.tab-nav {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 22px !important;
    padding: 10px 10px 0 !important;
    gap: 6px !important;
    margin: 0 0 18px 0 !important;
}

.tab-nav button {
    color: var(--muted) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 11px 14px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    background: transparent !important;
}

.tab-nav button:hover {
    color: var(--text) !important;
}

.tab-nav button.selected {
    color: #ffffff !important;
    background: #35A8E0 !important;
    box-shadow: 0 8px 18px rgba(53, 168, 224, 0.25) !important;
}

#chat-header-row {
    align-items: center !important;
    justify-content: space-between !important;
    gap: 14px !important;
    border-radius: 24px !important;
    padding: 18px 20px !important;
    margin-bottom: 14px !important;
}

.chat-title {
    font-size: 1.12rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em;
}

.chat-subtitle {
    color: var(--muted) !important;
    font-size: 0.88rem;
    margin-top: 4px;
}

.gradio-chatbot {
    border-radius: 24px !important;
    padding: 16px !important;
    overflow: hidden !important;
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    box-shadow: 0 20px 48px rgba(27, 46, 60, 0.06) !important;
    position: relative !important;
}

.gradio-chatbot > div {
    background-color: transparent !important;
}

/* Chatbot scrollable area spacing */
.gradio-chatbot .wrapper {
    padding: 8px !important;
}

/* Chatbot bubble wrap spacing */
.gradio-chatbot .bubble-wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

/* Message rows in newer Gradio */
.gradio-chatbot .message-row {
    margin-bottom: 12px !important;
    display: flex !important;
}

/* Avatar styling */
.gradio-chatbot .avatar-container {
    border: 2px solid var(--accent) !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
    background: var(--bg-2) !important;
    overflow: hidden !important;
}

/* Message base styling */
.gradio-chatbot .message {
    border-radius: 20px !important;
    padding: 14px 18px !important;
    line-height: 1.6 !important;
    font-size: 0.96rem !important;
    max-width: 85% !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* USER Bubble: Solid Sky Blue */
.user,
.message .user,
.message-wrap .user,
[data-testid="user-message"] {
    background: #35A8E0 !important;
    color: #070913 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 8px 24px rgba(53, 168, 224, 0.12) !important;
    font-weight: 500 !important;
    border-bottom-right-radius: 4px !important;
}

.user *, .message .user *, [data-testid="user-message"] * {
    color: #070913 !important;
}

.user:hover {
    box-shadow: 0 12px 30px rgba(53, 168, 224, 0.22) !important;
    transform: translateY(-1px) !important;
}

/* BOT / ASSISTANT Bubble: Warm Cream */
.bot,
.assistant,
.message .bot,
.message .assistant,
.message-wrap .bot,
.message-wrap .assistant,
[data-testid="bot-message"] {
    background: #070913 !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-left: 4px solid var(--accent) !important;
    box-shadow: 0 4px 12px rgba(27, 46, 60, 0.04) !important;
    border-bottom-left-radius: 4px !important;
}

.bot *, .assistant *, .message .bot *, .message .assistant *, [data-testid="bot-message"] * {
    color: var(--text) !important;
}

.bot:hover, .assistant:hover {
    background: #F3ECE0 !important;
    border-color: var(--accent) !important;
    box-shadow: 0 6px 18px rgba(27, 46, 60, 0.06) !important;
}

/* Markdown formatting inside bubbles */
.gradio-chatbot .message a {
    color: var(--accent) !important;
    text-decoration: underline !important;
    font-weight: 600 !important;
}
.gradio-chatbot .message a:hover {
    color: var(--accent-3) !important;
}

.gradio-chatbot .message pre,
.gradio-chatbot .message code {
    background: rgba(53, 168, 224, 0.05) !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    padding: 4px 6px !important;
}
.gradio-chatbot .message pre {
    padding: 10px !important;
    margin-top: 8px !important;
    margin-bottom: 8px !important;
    overflow-x: auto !important;
}

#input-container-row {
    align-items: center !important;
    gap: 12px !important;
    border-radius: 28px !important;
    padding: 6px 8px 6px 20px !important;
    margin-top: 18px !important;
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    box-shadow: 0 12px 30px rgba(27, 46, 60, 0.06) !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}

#input-container-row:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 12px 30px rgba(27, 46, 60, 0.06), 0 0 14px rgba(53, 168, 224, 0.15) !important;
}

#msg-input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

#msg-input, #msg-input label, #msg-input .container, #msg-input .scroll-hide, #msg-input > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    width: 100% !important;
}

#msg-input input, #msg-input textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-size: 1.02rem !important;
    min-height: 48px !important;
    padding: 12px 0 !important;
    outline: none !important;
}

#msg-input input::placeholder, #msg-input textarea::placeholder {
    color: var(--muted) !important;
}

#submit-btn,
#search-btn,
#upload-btn,
#save-btn,
#refresh-stats-btn,
#reset-db-btn,
#reload-sample-btn {
    border: none !important;
    border-radius: 999px !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

#submit-btn {
    background: #35A8E0 !important;
    color: #ffffff !important;
    min-width: 100px !important;
    height: 44px !important;
    box-shadow: 0 4px 12px rgba(53, 168, 224, 0.2) !important;
}

#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(53, 168, 224, 0.35) !important;
    filter: brightness(1.05) !important;
}

#submit-btn:active {
    transform: translateY(0) !important;
}

#clear-chat-btn {
    background: rgba(254, 95, 85, 0.06) !important;
    color: var(--danger) !important;
    border: 1px solid rgba(254, 95, 85, 0.16) !important;
    border-radius: 999px !important;
    min-width: 92px !important;
    height: 44px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
}

#clear-chat-btn:hover {
    background: var(--danger) !important;
    color: #ffffff !important;
    border-color: var(--danger) !important;
    transform: translateY(-1px) !important;
}

#mode-select {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    padding: 4px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}

#mode-select .wrap, #mode-select .radio-group, #mode-select > div {
    display: inline-flex !important;
    flex-direction: row !important;
    background: transparent !important;
    border: none !important;
    gap: 4px !important;
    width: auto !important;
}

#mode-select label {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 8px 14px !important;
    border-radius: 999px !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    background: transparent !important;
    border: none !important;
    margin: 0 !important;
}

#mode-select label:hover {
    color: var(--text) !important;
}

#mode-select label:has(input[type="radio"]:checked),
#mode-select label.selected {
    background: #35A8E0 !important;
    color: #ffffff !important;
}

#mode-select label input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
}

.settings-box {
    border-radius: 24px !important;
    padding: 22px !important;
}

.settings-box label,
.settings-box .markdown,
.settings-box p {
    color: var(--text) !important;
}

.settings-box input,
.settings-box textarea,
.settings-box select {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    border-radius: 14px !important;
}

.settings-box input:focus,
.settings-box textarea:focus,
.settings-box select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(53, 168, 224, 0.12) !important;
}

.gradio-dataframe,
.gradio-file,
.gradio-dropdown,
.gradio-textbox {
    border-radius: 18px !important;
}

/* Premium Light-Theme Radio Chip styling overrides */
.gradio-radio,
.gradio-radio .wrap,
.gradio-radio .radio-group {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    display: inline-flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    width: 100% !important;
}

.gradio-radio label,
.gradio-radio .radio-group label,
#provider-choice label {
    background: #070913 !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #FFFFFF !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.18s ease !important;
}

.gradio-radio label:hover,
#provider-choice label:hover {
    border-color: var(--accent) !important;
    background: #ffffff !important;
}

.gradio-radio label.selected,
.gradio-radio label:has(input[type="radio"]:checked),
#provider-choice label.selected,
#provider-choice label:has(input[type="radio"]:checked) {
    background: #35A8E0 !important;
    color: #ffffff !important;
    border-color: #35A8E0 !important;
}

.gradio-radio label span,
#provider-choice label span {
    color: inherit !important;
}

.info-card {
    border-radius: 24px !important;
    padding: 22px !important;
}

.info-card h3 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    margin-bottom: 12px !important;
    color: var(--text) !important;
}

.info-card ul {
    list-style: none !important;
}

.info-card li {
    color: var(--muted) !important;
    line-height: 1.6 !important;
    margin-bottom: 10px !important;
}

@media (max-width: 1100px) {
    .top-stats {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 760px) {
    .app-shell {
        padding: 18px 14px 32px;
    }

    .hero-band {
        padding: 20px 18px;
        border-radius: 22px;
    }

    #chat-header-row,
    #input-container-row {
        border-radius: 18px !important;
    }

    #input-container-row {
        flex-wrap: wrap !important;
    }

    #submit-btn,
    #clear-chat-btn {
        width: calc(50% - 5px) !important;
        min-width: 0 !important;
    }
}

"""

APP_HERO_HTML = """
<div class="hero-band">
  <div class="eyebrow">CricRAG AI Studio</div>
  <div class="hero-title">A premium <span>cricket intelligence</span> workspace</div>
  <div class="hero-copy">
    Search IPL history, compare players, inspect records, and route reasoning through local tools or Gemini.
    The experience is optimized for fast scanning, clean hierarchy, and minimal friction on desktop and mobile.
  </div>
  <div class="hero-meta">
    <div class="meta-pill"><strong>Offline-first</strong> cricket knowledge</div>
    <div class="meta-pill"><strong>MCP</strong> tool routing</div>
    <div class="meta-pill"><strong>Live</strong> Gemini / Ollama options</div>
  </div>
</div>
"""

APP_STATS_HTML = """
<div class="top-stats">
  <div class="stat-tile">
    <div class="stat-kicker">Knowledge Base</div>
    <div class="stat-number">1,000+</div>
    <div class="stat-text">Curated IPL facts, season summaries, player profiles, and rules.</div>
  </div>
  <div class="stat-tile">
    <div class="stat-kicker">Core Modes</div>
    <div class="stat-number">2</div>
    <div class="stat-text">Standard RAG and MCP Agent Mode with a shared visual system.</div>
  </div>
  <div class="stat-tile">
    <div class="stat-kicker">Response Paths</div>
    <div class="stat-number">3</div>
    <div class="stat-text">Offline simulator, Gemini, and Ollama with automatic fallback logic.</div>
  </div>
</div>
"""

APP_CHAT_BANNER_HTML = """
<div class="chat-banner" style="padding:14px 18px;border-radius:12px;margin-bottom:12px;background:rgba(53, 168, 224, 0.08);border:1px solid var(--line);">
    <strong style="font-size:0.95rem;color:var(--text);">CricRAG Assistant</strong>
    <div style="font-size:0.86rem;color:var(--muted);">Ask about players, seasons, records, or rules — powered by local DB and LLM fallbacks.</div>
</div>
"""

APP_RAIL_HTML = """
<div class="rail-panel" style="padding:12px;">
    <div style="font-weight:800;margin-bottom:8px;color:var(--text);">Quick Tools</div>
    <div style="display:flex;flex-direction:column;gap:8px;">
        <div style="font-size:0.92rem;color:var(--muted);">• Player stats lookup</div>
        <div style="font-size:0.92rem;color:var(--muted);">• Season summaries</div>
        <div style="font-size:0.92rem;color:var(--muted);">• Rules & regulations</div>
    </div>
</div>
"""

QUICK_PROMPTS = [
        ("Who is MS Dhoni?", "MS Dhoni career summary and IPL achievements"),
        ("Who has the most runs in IPL?", "All-time leading run scorers in IPL history"),
        ("Explain powerplay rules", "Powerplay rules in IPL and fielding restrictions"),
        ("Give 2023 IPL winner", "Who won the IPL 2023 season and key facts"),
]

ACCESSIBILITY_JS = ""

def build_app():
    saved_settings = load_settings()
    
    with gr.Blocks(title="CricRAG AI Studio") as demo:
        # Settings state
        settings_state = gr.State(saved_settings)

        gr.HTML(APP_HERO_HTML)
        gr.HTML(APP_STATS_HTML)
        
        with gr.Tabs() as tabs:
            # TAB 1: Chat Dashboard
            with gr.Tab("💬 CricRAG Assistant"):
                with gr.Row(elem_id="chat-header-row"):
                    gr.HTML("<div class='chat-title'>🏏 CricRAG Chat Stream</div><div class='chat-subtitle'>Premium cricket assistant</div>")
                    mode_select = gr.Radio(
                        choices=["Standard RAG", "MCP Agent Mode"],
                        value="MCP Agent Mode",
                        show_label=False,
                        container=False,
                        elem_id="mode-select"
                    )

                # Chat stage + Rail
                with gr.Row(elem_classes="assistant-grid"):
                    with gr.Column(elem_classes="assistant-stage"):
                        gr.HTML(APP_CHAT_BANNER_HTML)
                        chatbot = gr.Chatbot(
                            height=580,
                            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=cric")
                        )

                        with gr.Row(equal_height=False, elem_id="input-container-row"):
                            msg_input = gr.Textbox(
                                placeholder="Ask about IPL scores, player stats, records...",
                                show_label=False,
                                scale=8,
                                elem_id="msg-input",
                                container=False
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1, elem_id="submit-btn")
                            clear_chat = gr.Button("🗑️ Clear", variant="secondary", scale=1, elem_id="clear-chat-btn")

                    with gr.Column(elem_classes="insight-rail"):
                        gr.HTML(APP_RAIL_HTML)
                        # Quick prompt strip
                        prompt_buttons = []
                        for label, _desc in QUICK_PROMPTS:
                            prompt_buttons.append(gr.Button(label, variant="secondary"))
                        # Wire quick prompts to fill input
                        for i, btn in enumerate(prompt_buttons):
                            desc = QUICK_PROMPTS[i][1]
                            btn.click(lambda d=desc: d, inputs=[], outputs=[msg_input])
                            
                        # Accordion removed per user request
            
            # TAB 2: Semantic search explorer
            with gr.Tab("🔍 Semantic Explorer"):
                gr.Markdown("### Direct Vector Database Query")
                with gr.Row():
                    search_input = gr.Textbox(placeholder="Enter keyword or question to search the vector index...", label="Semantic Query", elem_id="search-input")
                    search_btn = gr.Button("Retrieve Chunks", variant="primary", elem_id="search-btn")
                    
                explorer_results = gr.Dataframe(
                    headers=["Source File", "Chunk Text", "Cosine Similarity", "Chunk Index"],
                    datatype=["str", "str", "str", "number"],
                    row_count=5,
                    wrap=False
                )
                
                search_btn.click(
                    fn=search_explorer_ui,
                    inputs=[search_input],
                    outputs=[explorer_results]
                )
                
            # TAB 3: Database Manager
            with gr.Tab("📂 Knowledge Ingestion"):
                gr.Markdown("### Manage CricRAG Vector Storage")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        db_stats_md = gr.Markdown(value="Loading statistics...")
                        refresh_stats_btn = gr.Button("Refresh Stats", variant="secondary", elem_id="refresh-stats-btn")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### Upload Cricket Documents (PDF, TXT, MD)")
                        uploader = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            label="Drag and drop or select files",
                            elem_id="file-uploader"
                        )
                        upload_btn = gr.Button("Ingest Documents", variant="primary", elem_id="upload-btn")
                        upload_status = gr.Markdown("")
                        
                with gr.Row():
                    reset_db_btn = gr.Button("Reset/Clear Vector Database", variant="stop", elem_id="reset-db-btn")
                    reload_sample_btn = gr.Button("Re-load Preloaded Cricket Database", variant="secondary", elem_id="reload-sample-btn")
                
                # Ingestion triggers
                upload_btn.click(
                    fn=upload_files,
                    inputs=[uploader],
                    outputs=[upload_status, db_stats_md]
                )
                
                reset_db_btn.click(
                    fn=clear_db_ui,
                    inputs=[],
                    outputs=[upload_status, db_stats_md]
                )
                
                def reload_sample():
                    global rag_engine
                    if rag_engine:
                        rag_engine.clear_database()
                        rag_engine.ingest_directory(os.path.join(BASE_DIR, "knowledge_base"))
                        return "Preloaded dataset loaded successfully!", get_stats_ui()
                    return "Engine not initialized.", ""
                    
                reload_sample_btn.click(
                    fn=reload_sample,
                    inputs=[],
                    outputs=[upload_status, db_stats_md]
                )
                
                refresh_stats_btn.click(
                    fn=get_stats_ui,
                    inputs=[],
                    outputs=[db_stats_md]
                )

            # TAB 4: Settings Panel
            with gr.Tab("⚙️ System Configuration"):
                with gr.Group(elem_classes="settings-box"):
                    gr.Markdown("### LLM Reasoning Configurations")
                    
                    provider_choice = gr.Radio(
                        choices=[
                            "Offline Simulator (Pre-compiled & Heuristics)",
                            "Google Gemini API",
                            "Local Ollama"
                        ],
                        value=saved_settings["provider"],
                        label="LLM Provider",
                        elem_id="provider-choice"
                    )
                    
                    ollama_input = gr.Textbox(
                        value=saved_settings["ollama_endpoint"],
                        placeholder="http://localhost:11434",
                        label="Local Ollama URL (Required for Ollama)",
                        elem_id="ollama-input"
                    )
                    
                    save_btn = gr.Button("Save Configurations", variant="primary", elem_id="save-btn")
                    save_status = gr.Markdown("")
                    
                    # Update State and Save settings
                    def apply_save_settings(provider, o_url):
                        msg = save_settings(provider, o_url)
                        new_state = {
                            "provider": provider,
                            "ollama_endpoint": o_url
                        }
                        return new_state, msg
                        
                    save_btn.click(
                        fn=apply_save_settings,
                        inputs=[provider_choice, ollama_input],
                        outputs=[settings_state, save_status]
                    )

        # Hidden component to satisfy callback mappings
        with gr.Row(visible=False):
            thought_output = gr.Markdown(
                value="*Ask a question in MCP Agent Mode to see the step-by-step reasoning process, tool calls, and execution logs.*",
                visible=False
            )

        # Wire up chat
        submit_btn.click(
            fn=chat_response,
            inputs=[msg_input, chatbot, mode_select, settings_state],
            outputs=[msg_input, chatbot, thought_output]
        )
        
        msg_input.submit(
            fn=chat_response,
            inputs=[msg_input, chatbot, mode_select, settings_state],
            outputs=[msg_input, chatbot, thought_output]
        )
        
        clear_chat.click(
            lambda: ("", [], "*Ask a question in MCP Agent Mode to see the step-by-step reasoning process, tool calls, and execution logs.*"),
            outputs=[msg_input, chatbot, thought_output]
        )

        # Load initial stats on layout completion
        demo.load(
            fn=get_stats_ui,
            inputs=[],
            outputs=[db_stats_md]
        )
        
    return demo

# Create unified FastAPI app
demo_app = build_app()
app = FastAPI()

from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    mode: str = "MCP Agent Mode"

class SettingsRequest(BaseModel):
    provider: str
    ollama_endpoint: Optional[str] = "http://localhost:11434"

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    settings = load_settings()
    # Call the chat_response logic
    new_input, new_history, thought_log = chat_response(req.message, req.history, req.mode, settings)
    return {
        "message": new_input,
        "history": new_history,
        "thought_log": thought_log
    }

@app.get("/api/settings")
def api_get_settings():
    return load_settings()

@app.post("/api/settings")
def api_post_settings(req: SettingsRequest):
    msg = save_settings(req.provider, req.ollama_endpoint)
    return {
        "status": "success",
        "message": msg
    }

from fastapi.responses import JSONResponse, Response

@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/manifest.json")
def get_manifest():
    manifest = {
        "short_name": "CricRAG",
        "name": "CricRAG AI Assistant",
        "icons": [],
        "start_url": "/",
        "background_color": "#0b0f19",
        "theme_color": "#38bdf8",
        "display": "standalone"
    }
    return JSONResponse(content=manifest)

@app.get("/favicon.ico")
def get_favicon():
    return Response(status_code=204)

@app.get("/chat_interface/gradio_api/mcp/schema")
def get_mcp_schema():
    return JSONResponse(content={"tools": [], "resources": [], "prompts": []})


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    return JSONResponse(content={})

# Mount Gradio at /chat_interface
app = gr.mount_gradio_app(
    app, 
    demo_app, 
    path="/chat_interface",
    css=CUSTOM_CSS
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7861))
    print(f"Starting Unified CricRAG FastAPI + Gradio Web Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60, log_level="debug")
