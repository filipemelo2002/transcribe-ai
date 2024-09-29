import os
from typing import  AnyStr
import torch
from pipeline import load_pipeline_from_pretrained

class Diarization:
  def __init__(self, token: AnyStr):
    # self.pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=token)
    self.pipeline = load_pipeline_from_pretrained(token)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.pipeline.to(device)

  
  def process_file(self, file, out_path):
    diarization = self.pipeline(file)
    
    os.makedirs(os.path.dirname(f'{out_path}.rttm'), exist_ok=True)
    
    with open(f'{out_path}.rttm', 'w') as out_file:
        diarization.write_rttm(out_file)
    