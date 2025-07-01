import os
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import faiss

from elevenlabs.client import ElevenLabs
# from elevenlabs import play
from pydub import AudioSegment
from pydub.playback import play
import io

from google import genai
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core import load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext

load_dotenv()

class AudioBot:
    def __init__(self):
        self.transcript_file = "data/audio_transcript.txt"
        self.model = "gemini-2.5-flash"
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)
        # self.full_transcript = []
        self._setup_llama_index()

    def _setup_llama_index(self):
        Settings.llm = Gemini(model=self.model, api_key=self.gemini_api_key)
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="text-embedding-004",
            api_key=self.gemini_api_key,
            embed_batch_size=100
        )
        Settings.node_parser = SimpleNodeParser.from_defaults()

    def record_audio(self, duration=5, sample_rate=44100):
        print("Recording...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            write(temp_audio.name, sample_rate, recording)
            return temp_audio.name

    def transcribe_audio(self, file_path):
        uploaded_file = self.client.files.upload(file=file_path)
        prompt = "Please transcribe this audio file and return clean, readable text."
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, uploaded_file]
        )
        return response.text.strip()

    def build_query_engine(self):
        persist_dir = "storage"
        os.makedirs(persist_dir, exist_ok=True)

        faiss_index_path = os.path.join(persist_dir, "faiss.index")
        docstore_path = os.path.join(persist_dir, "docstore.json")

        if os.path.exists(faiss_index_path) and os.path.exists(docstore_path):
            print("[INFO] Loading full index from storage...")
            faiss_index = faiss.read_index(faiss_index_path)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=persist_dir
            )
            index = load_index_from_storage(storage_context)
        else:
            print("[INFO] Creating new index...")
            documents = SimpleDirectoryReader("data").load_data()
            nodes = Settings.node_parser.get_nodes_from_documents(documents)

            # Create FAISS vector index
            embed_dim = len(Settings.embed_model.get_text_embedding("test"))
            faiss_index = faiss.IndexFlatL2(embed_dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(nodes, storage_context=storage_context)
            index.storage_context.persist()  # Persist the whole index

            faiss.write_index(faiss_index, faiss_index_path)
            print("[INFO] Index persisted to disk.")

        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.TREE_SUMMARIZE)
        return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    def generate_audio_response(self, text):
        elevenlabs = ElevenLabs(api_key=self.elevenlabs_api_key)
        audio = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        # play(audio)
        audio_bytes = b"".join(audio)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        play(audio_segment)

    def run(self):
        while True:
            print("\nSay something (or say 'exit' to quit)...")
            audio_path = self.record_audio()
            transcribed_text = self.transcribe_audio(audio_path)

            if transcribed_text.lower() in ["exit", "quit"]:
                print("Exiting conversation.")
                break

            query_engine = self.build_query_engine()

            print(f"\n[You]: {transcribed_text}")
            response = query_engine.query(transcribed_text)

            print(f"\n[Assistant]: {response.response}")
            self.generate_audio_response(response.response[:250])

if __name__ == "__main__":
    bot = AudioBot()
    bot.run()