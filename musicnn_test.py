from pathlib import Path
from musicnn.tagger import top_tags

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

def validate_musicnn(dataset_path):
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
                # Get all tags
                tags = top_tags(str(audio_file), model='MTT_musicnn', topN=10)
                
                # Find first tag that's in any genre group
                predicted_genre = 'unknown'
                for tag in tags:
                    tag_name = tag.lower()
                    # Check if tag is in any of our genre groups
                    for group_genres in GENRE_GROUPS.values():
                        if tag_name in group_genres:
                            predicted_genre = tag_name
                            break
                    if predicted_genre != 'unknown':
                        break
                
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
    validate_musicnn(dataset_path)