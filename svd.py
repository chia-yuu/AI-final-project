import dataset
import numpy
import heapq
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

def svd(data, uid, watched_history):
    # choose best parameters: 
    # uses GridSearchCV to find the best parameters for the SVD model (number of epochs, learning rate, and number of factors).
    param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "n_factors": [50, 100,150]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, joblib_verbose=1, n_jobs=2)
    gs.fit(data)
    algo = gs.best_estimator["rmse"]
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    predictions = algo.test(trainset.build_testset())
    accuracy.rmse(predictions)

    # trains the SVD model with the best parameters
    best_epochs = gs.best_params["rmse"]["n_epochs"] 
    best_factors = gs.best_params["rmse"]["n_factors"]
    svd_sol_best = SVD(verbose=True, n_epochs=best_epochs, n_factors = best_factors) 
    cross_validate(svd_sol_best, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # makes predictions based on the given movie for the given user
    score = []
    for iid in range(1682):
        pred_best = svd_sol_best.predict(uid, iid, r_ui=9, verbose=True)
        score.append(pred_best.est)

    # choose the top 5 similar movie
    top_5_indices = sorted(range(len(score)), key=lambda i: (-score[i], i), reverse=True)[:5]

    #Output
    for i in top_5_indices:
        name = dataset.find_movie_name(i)
        est_score = score[i]
        print("%60s || Score = %.4f" %(name, est_score))

        
def get_movie_genres(movie_id, movies):
    '''
    Record the genres to which the given movie belongs
    '''
    i = 0
    genres = [] # genres

    while(i < 1682):
        if (int(movies[i][0]) == movie_id): # find the data for the given movie_id
            # movies[i][5]~movies[i][23] are types of movies
            j = 5
            while(j <= 23):
                # If movies[i][j] == 1, then the movie is belong to this type. So append to genres.
                # Otherwise, just ignore.
                if(int(movies[i][j]) == 1):
                    genres.append(j)
                j = j + 1
            break
        i = i + 1
    return genres

def recommend_movies(movie_id, movies):
    """
    Record movies with similar genres based on the input movie ID.
    """
    input_movie_genres = get_movie_genres(movie_id, movies) # Retrieve the genres of the given movie.
    re_movie_id = [] # Record ids of movies with similar genres.
    i = 0
    while(i < 1682):
        if(int(movies[i][0]) != movie_id): # Avoid recording the given movie itself.
            j = 0
            while(j < len(input_movie_genres)):
                idx = input_movie_genres[j] # The idx-th genre
                # If the i-th movie belongs to the idx-th genre(int(movies[i][idx]) == 1),
                # it means that the i-th movie and the given movie belong to the same genre.
                # So, append to re_movie_id 
                if(int(movies[i][idx]) == 1):
                    re_movie_id.append(i + 1)
                    break
                j = j + 1
        i = i + 1
    return re_movie_id

def svd_main():
    # load data
    rating = dataset.load_data()    # user id | movie id | rating | timestamp
    movies = dataset.load_movie()   # movie id | movie title | release date | video release date |IMDb URL | 類型(19項)
    watched_history = []

    # input 
    while(True):
        # input
        watched_movie_name = input("Please enter a movie name (enter 0 to exit): ")
        if(watched_movie_name=="0"): break
        user_id = input("Please enter your id: ")
        watched_movie_id = dataset.find_movie_id(watched_movie_name)
        if(watched_movie_id==-1): continue
        watched_history.append(watched_movie_id)

        # Choose movies that belong to the same genres  as given movie to traing svd.
        train_movie = recommend_movies(watched_movie_id + 1, movies)
        train_rating = []
        for r in rating:
            if(r[0] == user_id):
                for index in train_movie:
                    if(int(index) == int(r[1])):
                        train_rating.append(r)

        # If there are no movies with the same genre as the given movie, 
        # just use origin rating to train svd.
        if(len(train_rating) == 0):
            for r in rating:
                if(int(r[0]) == user_id):
                    train_rating.append(r)

        # Create a Reader object with a rating scale of 1 to 5
        reader = Reader(rating_scale=(1, 5))

        # Create a Dataset object from the train_rating data using the Reader object
        data = Dataset.load_from_df(pd.DataFrame({'user_id': [int(x[0]) for x in train_rating], 
                                            'anime_id': [int(x[1]) for x in train_rating],
                                            'rating': [int(x[2]) for x in train_rating]}), 
                                    reader)
        svd(data, user_id, watched_history)

if __name__ == "__main__":
    svd_main()
