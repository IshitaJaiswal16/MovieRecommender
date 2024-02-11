import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##DATA COLLECTION AND PRE-PROCESSING

#loading the data from the csv file t a pandas dataframe
movies_data=pd.read_csv(r'C:\Users\ishit\Downloads\movies.csv')

#selecting the relevant features for recommendation
selected_features=['genres','keywords','original_language','overview','cast','director']

#replacing the null values with null string
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna(' ')

#combining the selected features
combined_features=movies_data['genres']+movies_data['keywords']+movies_data['original_language']+movies_data['overview']+movies_data['cast']+movies_data['director']

#converting the text data to feature vectors
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)

#getting the similarity scores using cosine similarity
similarity=cosine_similarity(feature_vectors)



#function to recommend movies
def recommending_movie(movie_name):

    # finding the index of the movie title
    movie_index = movies_data[movies_data.title == movie_name]['index'].values[0]

    # getting a list of similar movies
    similarity_score = list(enumerate(similarity[movie_index]))

    # sorting movie based on similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:]
    movies_recommended=[]
    i=0
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if (i < 10):
            movies_recommended.append(title_from_index)
            i += 1
    return movies_recommended




#PRESENTING IT ON WEB

st.title("MOVIE RECOMMENDATION SYSTEM")
movies_list=movies_data['title'].tolist()

#getting the movie name from the user
Movie_name = st.selectbox("Select movie from dropdown", movies_list)

#displaying the results on the screen
if st.button("Search"):
    movies= recommending_movie(Movie_name)

    col1, col2 = st.columns(2)
    with col1:
        for i in range(5):
            st.text(movies[i])
    with col2:
        for i in range(5, 10):
            st.text(movies[i])
