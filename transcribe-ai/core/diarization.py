import os
from typing import  AnyStr, List
import torch
from .pipeline import load_pipeline_from_pretrained
from .diarization_segment import DiarizationSegment
import json

class Diarization:
    def __init__(self, token: AnyStr):
        self.pipeline = load_pipeline_from_pretrained(token)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline.to(device)

    def process_file(self, file) -> List[DiarizationSegment]:
        diarization = self.pipeline(file)
        
        processed_diarization = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = DiarizationSegment(start=f"{turn.start:.3f}", end=f"{turn.end:.3f}", speaker_id=speaker)
            processed_diarization.append(segment)
        
        return processed_diarization

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
        
        with open('./diarization-output.json', 'w') as out_file:
            json.dump({"diarization": processed_diarization}, out_file)
        
        
    