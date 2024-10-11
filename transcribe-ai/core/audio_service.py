from pydub import AudioSegment
import numpy as np

class AudioService:
    def extract_segment(self, file, start: float, end: float):
        audio_file = AudioSegment.from_file(file, format="wav")
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