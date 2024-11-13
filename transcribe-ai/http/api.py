from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ..core.diarization import Diarization
from ..core.transcribe import Transcribe
from ..core.audio_service import AudioService
import os
from dotenv import load_dotenv
load_dotenv()

token = os.environ['HF_AUTH_TOKEN']
diarization_service = Diarization(token=token)
audio_service = AudioService()
transcribe_service = Transcribe()


app = FastAPI()
origins = ['*']

app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=origins,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post("/transcribe")
async def transcribe_file(file: UploadFile):
    sound = audio_service.load_audio_file(file=file)
    diarization_segments = diarization_service.process_file(sound)
    last_end = 0
    for  segment in diarization_segments:
        audio_file = audio_service.extract_segment(file=sound, start=float(segment.start) * 1000, end=float(segment.end)*1000)
        transcriptions = transcribe_service.transcribe_audio(file=audio_file, append_time=last_end)
        segment.transcriptions = transcriptions
        last_end = float(segment.end)

    os.remove(sound)
    return diarization_segments

@app.get("/hello")
def hello_world():
        return {'message': 'Hello world from Transcribe-AI'}