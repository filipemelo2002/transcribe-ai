import os
from dotenv import load_dotenv
from diarization import Diarization


load_dotenv()
dirname = os.path.dirname(__file__)

diarization = Diarization(token=os.environ['HF_AUTH_TOKEN'])