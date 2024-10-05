from fastapi import FastAPI, UploadFile
from core.diarization import Diarization
import os
from dotenv import load_dotenv
load_dotenv()

token = os.environ['HF_AUTH_TOKEN']
model = Diarization(token=token)

app = FastAPI()

@app.post("/transcribe")
def transcribe_file(audio: UploadFile):
    print(f"incoming file {audio}")
    return model.process_file(audio.file)     

@app.get("/hello")
def hello_world():
        return {'message': 'Hello world from Transcribe-AI'}