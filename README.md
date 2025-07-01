# AudioBot

## Prerequisites
- [FFmpeg Installed](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip)
- Add bin folder of ffmpeg to PATH

---

## Getting Started
### 1. Clone the repository
```bash
git clone <repository-url>
cd AudioBot
```
#### 2. Create Virtual Environment
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 4. Set up environment variables (.env file):
```bash
GEMINI_API_KEY=<your_google_api_key>
ELEVENLABS_API_KEY=<your_elevenlabs_api_key>
```
#### 5. Run the bot:
```
python main.py
```

---

## Technology Stack
- STT: Gemini 2.5 Flash (google-genai)
- RAG: LlamaIndex
- TTS: ElevenLabs and PyDub

---

## Features
### Add Required Data
- Add your data as .txt format in `data` folder
### Gemini STT
- Recorded Audio is transcribed using `gemini-2.5-flash` model
### RAG Pipeline
- Data Ingestion and Parsing: used SimpleDirectoryReader and node_parser
- Embedding & Indexing: used Google GenAI and FAISS vector index
- Storage Management: used StorageContext 
- Retrieval: used VectorIndexRetriever
### Eleven Labs TTS
- Converts LLM text response to audio
- Used PyDub for optimization
