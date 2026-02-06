import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib  # Standard Python library for finding close matches (Typos)

# ==========================================
# 1. MODEL TRAINING
# ==========================================
def train_model(df):
    """
    Vectorizes genres and calculates the cosine similarity matrix.
    """
    cv = CountVectorizer()
    
    # Fill NaN values to avoid errors
    df['genre'] = df['genre'].fillna('') 
    
    # Create the matrix based on Genre
    count_matrix = cv.fit_transform(df['genre'])
    
    # Calculate Cosine Similarity
    similarity_matrix = cosine_similarity(count_matrix)
    
    return similarity_matrix

# ==========================================
# 2. HELPER: FIND CLOSEST MOVIE TITLE
# ==========================================
def get_movie_from_input(user_input, all_titles):
    """
    Finds the closest movie title in the database to what the user typed.
    Example: User types "Batmn" -> Returns "The Dark Knight" or "Batman"
    """
    # specific_match checks if the user input is contained in any title (e.g., "Matrix" in "The Matrix")
    found_match = difflib.get_close_matches(user_input, all_titles, n=1, cutoff=0.4)
    
    if not found_match:
        # Try a simpler containment search
        for title in all_titles:
            if user_input.lower() in title.lower():
                return title
        return None
    
    return found_match[0]

# ==========================================
# 3. RECOMMENDATION LOGIC
# ==========================================
def recommend_movie(movie_user_likes, df, similarity_matrix):
    """
    Finds and prints movies similar to the input movie.
    """
    # Get the index of the movie from the dataframe
    try:
        movie_index = df[df.title == movie_user_likes].index[0]
    except IndexError:
        print("Movie not found in index.")
        return

    # Get similarity scores
    similar_movies = list(enumerate(similarity_matrix[movie_index]))

    # Sort movies based on similarity score (Highest first)
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    print(f"\n✅ Because you liked **{movie_user_likes}** ({df.iloc[movie_index]['genre']}):")
    print("-" * 40)
    
    count = 1
    for movie in sorted_movies:
        index = movie[0]
        title = df.iloc[index]['title']
        genre = df.iloc[index]['genre']
        
        # Skip the movie itself
        if title == movie_user_likes:
            continue
            
        print(f"{count}. {title:<40} | Genre: {genre}")
        count += 1
        if count > 5: # Limit to top 5 recommendations
            break
    print("-" * 40)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("   🎬 INTELLIGENT MOVIE RECOMMENDER")
    print("==========================================\n")

    # 1. Load Data
    try:
        df = pd.read_csv('movies.csv')
        print(f"[✔] Dataset loaded: {len(df)} movies.")
    except FileNotFoundError:
        print("[X] Error: 'movies.csv' not found.")
        exit()

    # 2. Train Model
    sim_matrix = train_model(df)
    print("[✔] Model trained successfully.\n")

    # 3. INTERACTIVE LOOP (Finding User Taste)
    all_titles = df['title'].tolist()
    
    while True:
        print("\nWhat kind of movie are you in the mood for?")
        user_input = input("👉 Enter a movie name (or 'q' to quit): ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Goodbye! Enjoy your movie! 🍿")
            break
            
        # Find the closest match to what the user typed
        closest_match = get_movie_from_input(user_input, all_titles)
        
        if closest_match:
            recommend_movie(closest_match, df, sim_matrix)
        else:
            print(f"❌ Sorry, we couldn't find a match for '{user_input}'. Try another one!")
            print("Tip: Check 'movies.csv' to see available movies.")