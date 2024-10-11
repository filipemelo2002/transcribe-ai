from fastapi import FastAPI, UploadFile
from core.diarization import Diarization
from core.transcribe import Transcribe
from core.audio_service import AudioService
import os
from dotenv import load_dotenv
load_dotenv()

token = os.environ['HF_AUTH_TOKEN']
diarization_service = Diarization(token=token)
audio_service = AudioService()
transcribe_service = Transcribe()


app = FastAPI()

@app.post("/transcribe")
async def transcribe_file(audio: UploadFile):
    diarization_segments = diarization_service.process_file(audio.file)
    for  segment in diarization_segments:
        audio_file = audio_service.extract_segment(file=audio.file, start=float(segment.start) * 1000, end=float(segment.end)*1000)
        transcriptions = transcribe_service.transcribe_audio(file=audio_file)
        segment.transcriptions = transcriptions
    return diarization_segments

@app.get("/hello")
def hello_world():
        return {'message': 'Hello world from Transcribe-AI'}