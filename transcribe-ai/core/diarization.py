import os
from typing import  AnyStr
import torch
from .pipeline import load_pipeline_from_pretrained
from pyannote.audio import Audio

class Diarization:
    def __init__(self, token: AnyStr):
        self.pipeline = load_pipeline_from_pretrained(token)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline.to(device)

    def process_file(self, file, out_path):
        diarization = self.pipeline(file)
        
        os.makedirs(os.path.dirname(f'{out_path}.rttm'), exist_ok=True)

        with open(f'{out_path}.rttm', 'w') as out_file:
            diarization.write_rttm(out_file)

    def process_request(self, file):
        
        diarization = self.pipeline(file) 
        
        processed_diarization = [
            {
                "speaker": speaker,
                "start": f"{turn.start:.3f}",
                "end": f"{turn.end:.3f}",
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        
        return {"diarization": processed_diarization}
        
        
    