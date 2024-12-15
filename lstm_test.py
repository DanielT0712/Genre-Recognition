from pathlib import Path
import logging
from keras.models import model_from_json
import numpy as np
import librosa

# Set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# group similar genres
GENRE_GROUPS = {
    # Rock and related genres
    'rock': {'rock', 'hard rock', 'punk', 'metal', 'heavy metal', 'soft rock'},
    
    # Electronic and dance music
    'electronic': {'electronic', 'electronica', 'techno', 'trance', 'house', 'electro', 'ambient', 'industrial'},
    
    # Classical and orchestral
    'classical': {'classical', 'baroque', 'orchestra', 'orchestral', 'opera', 'medieval', 'chamber'},
    
    # Jazz and blues related
    'jazz': {'jazz', 'jazzy', 'blues', 'funk', 'funky'},
    
    # Folk and acoustic
    'folk': {'folk', 'celtic', 'irish', 'acoustic', 'world'},
    
    # Urban and rhythm-based
    'hiphop': {'hip hop', 'rap', 'beats', 'urban'},
    
    # Pop and mainstream
    'pop': {'pop', 'disco', 'dance'},
    
    # Reggae and related
    'reggae': {'reggae', 'jungle'},
    
    # Country and western
    'country': {'country', 'western', 'banjo'}
}

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

def get_genre_group(genre):
    """Find which group a genre belongs to"""
    genre = genre.lower()
    for group, genres in GENRE_GROUPS.items():
        if genre in genres:
            return group
    return genre  # Return original genre if no group found

def are_genres_similar(genre1, genre2):
    """Check if two genres are similar based on genre groups"""
    group1 = get_genre_group(genre1)
    group2 = get_genre_group(genre2)
    return group1 == group2

def validate_lstm(dataset_path, model):

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
    # Track stats per genre
    genre_stats = {}
    
    # Go through each genre folder
    for genre_dir in Path(dataset_path).iterdir():
        if not genre_dir.is_dir():
            continue
            
        true_genre = genre_dir.name.lower()
        print(f"\nProcessing {true_genre} folder...")
        genre_stats[true_genre] = {
            'correct': 0,
            'similar': 0,
            'wrong': 0,
            'total': 0
        }
        
        # Test each audio file
        for audio_file in genre_dir.glob("*.wav"):
            try:
                # Get prediction
                features = extract_audio_features(str(audio_file))
                prediction = model.predict(features, verbose=0)
                predicted_genre = genre_list[np.argmax(prediction)].lower()
                
                # Update stats
                genre_stats[true_genre]['total'] += 1
                
                if predicted_genre == true_genre:
                    genre_stats[true_genre]['correct'] += 1
                elif are_genres_similar(predicted_genre, true_genre):
                    genre_stats[true_genre]['similar'] += 1
                else:
                    genre_stats[true_genre]['wrong'] += 1
                
                print(f"{audio_file.name}: Predicted {predicted_genre}, Actual {true_genre}")
                
            except Exception as e:
                print(f"Error with {audio_file}: {e}")
    
    # Print final results
    print("\nResults by Genre:")
    print("-" * 70)
    total_correct = 0
    total_similar = 0
    total_files = 0
    
    for genre in sorted(genre_stats):
        stats = genre_stats[genre]
        correct = stats['correct']
        similar = stats['similar']
        total = stats['total']
        
        exact_accuracy = (correct / total * 100) if total > 0 else 0
        similar_accuracy = (similar / total * 100) if total > 0 else 0
        
        print(f"{genre:10} : Exact: {exact_accuracy:6.2f}% Similar: {similar_accuracy:6.2f}% "
              f"({correct}/{similar}/{total} correct/similar/total)")
        
        total_correct += correct
        total_similar += similar
        total_files += total
    
    print("-" * 70)
    exact_overall = (total_correct / total_files * 100) if total_files > 0 else 0
    similar_overall = ((total_correct + total_similar) / total_files * 100) if total_files > 0 else 0
    print(f"Overall    : Exact: {exact_overall:6.2f}% Similar: {similar_overall:6.2f}% "
          f"({total_correct}/{total_similar}/{total_files} correct/similar/total)")

if __name__ == "__main__":
    dataset_path = "./genres_original"
    model = load_model("./weights/model.json", "./weights/model_weights.h5")
    validate_lstm(dataset_path, model)