import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 16000

def preprocess_audio(input_path, output_path):

    audio, sr = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True
    )

    audio = audio / np.max(np.abs(audio))

    sf.write(output_path, audio, TARGET_SR)

    return output_path