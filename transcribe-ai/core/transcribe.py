import torch
import whisper
from .transcription import Transcription


class Transcribe:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model("small", device = device)
    
    def transcribe_audio(self, file, language='en'):

        results = self.model.transcribe(audio=file, language=language, word_timestamps=True)
        transcriptions = []
        
        for segment in results['segments']:
            for word in segment['words']:
                transcriptions.append(Transcription(
                    word=word['word'],
                    start=word['start'],
                    end=word['end']
                ))
        
        return transcriptions
            
