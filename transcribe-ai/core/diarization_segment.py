from pydantic import BaseModel
from .transcription import Transcription
from typing import List

class DiarizationSegment(BaseModel):
    start: str
    end: str
    speaker_id: str
    transcriptions: List[Transcription] = []