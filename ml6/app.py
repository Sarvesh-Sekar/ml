import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    'title': ["Inception", "Interstellar", "The Matrix", "The Prestige", "The Dark Knight", "Memento"],
    'genre': ["Sci-Fi Thriller", "Sci-Fi Adventure", "Sci-Fi Action", "Mystery Thriller", "Action Thriller", "Mystery Thriller"],
    'description': [
        "A thief who steals corporate secrets through the use of dream sharing technology.",
        "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
        "Two magicians engage in a competitive rivalry to create the ultimate stage illusion.",
        "Batman raises the stakes in his war on crime with the help of Lieutenant Jim Gordon and District Attorney Harvey Dent.",
        "A man with short-term memory loss attempts to track down his wife's murderer."
    ]
}

# Create DataFrame
movies = pd.DataFrame(data)

# Combine genre and description into a single feature
movies['combined_features'] = movies['genre'] + " " + movies['description']

# Apply TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim, df=movies):
    # Get the index of the movie with the given title
    idx = df[df['title'] == title].index[0]

    # Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]

    # Get the indices of the top 3 most similar movies
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices]

# Test the recommendation function
print("Movies similar to 'Inception':")
print(recommend_movies('Inception'))

# Loop through each movie and print recommendations
for movie in movies['title']:
    print(f"Movies similar to '{movie}':")
    print(recommend_movies(movie))
    print("\n" + "-"*30 + "\n")
