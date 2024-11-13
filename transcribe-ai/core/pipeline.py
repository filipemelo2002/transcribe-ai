import os
from pathlib import Path
from pyannote.audio import Pipeline
dirname = os.path.dirname(__file__)


def load_pipeline_from_pretrained(token: str) -> Pipeline:
    path_to_config = Path(os.path.join(dirname, '../../models/pyannote_diarization_config.yaml'))
    
    print(f"Reading config file from {path_to_config}")
    
    pipeline = Pipeline.from_pretrained(path_to_config, use_auth_token=token)
    
    print("Pipeline successfully created")
    
    return pipeline