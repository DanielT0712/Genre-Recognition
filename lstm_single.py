# Updated imports to use tensorflow.keras instead of standalone keras
import librosa
import logging
import sys
import numpy as np
from keras.models import model_from_json
#keras==2.2.4, h5py==2.10.0
# Set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

genre_list = [
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
]

def load_model(model_path, weights_path):
    "Load the trained LSTM model from directory for genre classification"
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    return trained_model

# Rest of the functions remain the same
def extract_audio_features(file):
    "Extract audio features from an audio file for genre classification"
    timeseries_length = 128
    features = np.zeros((1, timeseries_length, 33), dtype=np.float64)

    y, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features

def get_genre(model, music_path):
    "Predict genre of music using a trained model"
    prediction = model.predict(extract_audio_features(music_path))
    predict_genre = genre_list[np.argmax(prediction)]
    return predict_genre

if __name__ == "__main__":
    PATH = sys.argv[1] if len(sys.argv) == 2 else "./sample/John Denver - Take Me Home, Country Roads (Official Audio).mp3"
    MODEL = load_model("./weights/model.json", "./weights/model_weights.h5")
    GENRE = get_genre(MODEL, PATH)
    print("Model predict: {}".format(GENRE))