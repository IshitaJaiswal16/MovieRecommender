import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##DATA COLLECTION AND PRE-PROCESSING

#loading the data from the csv file t a pandas dataframe
movies_data=pd.read_csv(r"C:\Users\ishit\Downloads\movies.csv")

#selecting the relevant features for recommendation
selected_features=['genres','keywords','original_language','overview','cast','director']

#replacing the null values with null string
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna('')

#combining the selected features
combined_features=movies_data['genres']+movies_data['keywords']+movies_data['original_language']+movies_data['overview']+movies_data['cast']+movies_data['director']

#converting the text data to feature vectors
vectorizer=TfidfVectorizer()
feature_vectors=vectorizer.fit_transform(combined_features)

#getting the similarity scores using cosine similarity
similarity=cosine_similarity(feature_vectors)


##GETTING THE MOVIE NAME FROM THE USER AND RECOMMENDING THE MOVIES

print('MOVIE RECOMMENDATION SYSYTEM')

#getting the movie name from the user
movie_name=input('Enter the movie name: ')

#creating a list with all the movie names given in the dataset
list_of_all_titles=movies_data['title'].tolist()

#finding the close match for the ,movie name given by the user
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
close_match=find_close_match[0]

#finding the index of the movie title
movie_index=movies_data[movies_data.title==close_match]['index'].values[0]

#getting a list of similar movies
similarity_score=list(enumerate(similarity[movie_index]))

#sorting movie based on similarity score
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

#printing the recommended movies
print('Movies suggested for you: \n')

i=1

for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if (i<11):
        print(title_from_index)
        i+=1
