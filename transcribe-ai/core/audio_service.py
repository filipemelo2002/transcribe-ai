import os
from pathlib import Path
from time import time
from pydub import AudioSegment
import numpy as np
dirname = os.path.dirname(__file__)

class AudioService:
    def extract_segment(self, file, start: float, end: float):
        audio_file = AudioSegment.from_file(file)
        audio = audio_file[start:end]
        
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.sample_width != 2:  
            audio = audio.set_sample_width(2)
        if audio.channels != 1:       # mono
            audio = audio.set_channels(1)        
        arr = np.array(audio.get_array_of_samples())
        
        arr = arr.astype(np.float32)/32768.0
        
        return arr

    def load_audio_file(self, file, resize=False):
        sound = None
        filename = file.filename
        file_extension = ''
        if filename.endswith('.mp3') or filename.endswith('.MP3'):
            sound = AudioSegment.from_mp3(file.file)
        elif filename.endswith('.wav') or filename.endswith('.WAV'):
            sound = AudioSegment.from_wav(file.file)
        elif filename.endswith('.ogg'):
            sound = AudioSegment.from_ogg(file.file)
        elif filename.endswith('.flac'):
            sound = AudioSegment.from_file(file.file, "flac")
        elif filename.endswith('.3gp'):
            sound = AudioSegment.from_file(file.file, "3gp")
        elif filename.endswith('.3g'):
            sound = AudioSegment.from_file(file.file, "3gp")
        elif filename.endswith('.mp4'):
            file_extension = '.mp4'
            sound = AudioSegment.from_file(file.file, 'mp4')

        sound = sound.set_frame_rate(16000)
        sound = sound.set_channels(1)
        sound = sound.set_sample_width(2)
        
        path_to_export = Path(os.path.join(dirname, f'../../tmp/{filename.replace(file_extension, "")}-{time()}.wav'))
        sound.export(path_to_export, format='wav')
        return path_to_export
