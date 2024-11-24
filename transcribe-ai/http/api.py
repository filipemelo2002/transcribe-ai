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
    diarized_segments = diarization_service.process_file(sound)
    response = []
    for  segment in diarized_segments:
        audio_file = audio_service.extract_segment(file=sound, start=float(segment.start) * 1000, end=float(segment.end)*1000)
        transcriptions = transcribe_service.transcribe_audio(file=audio_file, append_time=float(segment.start))
        segment.transcriptions = transcriptions
        if len(transcriptions) > 0:
            response.append(segment)

    os.remove(sound)
    return response

@app.get("/hello")
def hello_world():
        return {'message': 'Hello world from Transcribe-AI'}