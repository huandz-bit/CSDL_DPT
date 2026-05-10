import librosa
import numpy as np

from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()

def extract_features(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    # ===== MFCC =====
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    # ===== Pitch =====
    f0, _, _ = librosa.pyin(
        y,
        fmin=50,
        fmax=500
    )

    f0_mean = np.nanmean(f0)

    # ===== Spectral =====
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    # ===== Speaker Embedding =====
    wav = preprocess_wav(audio_path)

    embedding = encoder.embed_utterance(wav)

    embedding = embedding / np.linalg.norm(embedding)

    return {

        "mfcc_mean": mfcc_mean.tolist(),

        "pitch": {
            "f0_mean": float(f0_mean)
        },

        "spectral": {
            "centroid": float(spectral_centroid)
        },

        "speaker_embedding": embedding.tolist()
    }