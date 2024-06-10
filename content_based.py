# Input movie name, output 5 recommended movies
# User can watch multiple movies and recommend according to these movies
# Use cosine similarity to calculate the similarity between movies
# Randomness. If similarity are the same, then recommend the movie randomly, so it won't recommend the same movie every time by the movie ID.
import dataset
import heapq
import math
import random

def content_based():
    # load data
    movies = dataset.load_movie()   # movie id | movie title | release date | video release date |IMDb URL | 類型(19項)
    rating = dataset.load_data()

    user_movie = [0]*19     # record the overall type of the user's watched
    watched_history = []    # record the movies that user had watched, so we won't recommend the movie again
    avg_rate = 3.52986

    # construct user-movie matrix
    '''
    um_matrix is a 944 * 1683 matrix, record the rating of the movies by each user
    um_matrix[user][movie] = [rating]
    the mtx size is +1 because the movie id start from 1
    '''
    um_matrix = []
    # initialize
    for u in range(944):
        um_matrix.append([])
        for m in range(1683):
            um_matrix[u].append(0)
    # fill in value
    for r in rating:
        um_matrix[int(r[0])][int(r[1])] = int(r[2])

    # calculate similarity matrix (Cosine Similarity)
    '''
    similarity_matrix is a 1683 * 1683 matrix, record the similarity between each movie
    similarity_matrix[i][j] = similarity between movie i and movie j
    the last column (last row) is for the user's watched movie
    we calculate its movie type according to the user's watching history
    and construct the similarity between it and other movies (this part is done in the whiel loop)
    '''
    print("collaborative preparing dataset")
    similarity_matrix = []
    for i in range(1683):
        similarity_matrix.append([])
        for j in range(1683):
            if(i==1682 or j==1682):
                similarity_matrix[i].append(0)
                continue
            a, b, c = 0, 0, 0
            for k in range(5, 24):
                a += int(movies[i][k]) * int(movies[j][k])
                b += int(movies[i][k]) * int(movies[i][k])
                c += int(movies[j][k]) * int(movies[j][k])
            if(b*c==0): similarity = a      # avoid devide by zero
            else: similarity = a / (math.sqrt(b) * math.sqrt(c))
            similarity_matrix[i].append(similarity)

    while(True):
        # allow the user to watch multiple movies, and recommend movies based on these watched movies
        n_movie = 0     # count how many movies have the user watched
        sum = 0
        watched_history = []
        user_movie = [0]*19
        while(True):
            # input
            watched_movie_name = input("Please enter a movie name (enter 0 to calculate RMSE or enter 00 to exist): ")
            if(watched_movie_name == "00" or watched_movie_name == "0"): break
            watched_movie_id = dataset.find_movie_id(watched_movie_name)
            if(watched_movie_id==-1): continue
            n_movie +=1
            watched_history.append(watched_movie_id)

            # update user's prefer movie type
            for i in range(19):
                user_movie[i] = (user_movie[i] * (n_movie-1) + int(movies[watched_movie_id][i+5])) / n_movie
            # print(user_movie)

            # calculate similarity for the user movie
            for i in range(1682):
                a, b, c = 0, 0, 0
                for k in range(5, 24):
                    a += int(movies[i][k]) * user_movie[k-5]
                    b += int(movies[i][k]) * int(movies[i][k])
                    c += user_movie[k-5] * user_movie[k-5]
                if(b*c==0): similarity = a      # avoid devide by zero
                else: similarity = a / (math.sqrt(b) * math.sqrt(c))
                similarity_matrix[1682][i] = similarity

            # choose the top 5 similar movie
            pq = []
            for i in range(1682):
                # to make it a max heap (large value at the front), let similarity * (-1)
                if(i not in watched_history):
                    heapq.heappush(pq, (similarity_matrix[1682][i] * (-1), i))

            # output
            print("You may like the following 5 movies:")

            # if movies have the same similarity, then output a random movie
            n_output = 5
            while(n_output>0):
                same_s_movie = []
                tmp_s, tmp_id = heapq.heappop(pq)
                if(tmp_id==watched_movie_id):
                    i-=1
                    continue
                same_s_movie.append((tmp_s, tmp_id))
                while(True):
                    tt_s, tt_id = heapq.heappop(pq)
                    if(tt_s==tmp_s): same_s_movie.append((tt_s, tt_id))
                    else:
                        heapq.heappush(pq, (tt_s, tt_id))
                        break
                if(len(same_s_movie)<n_output):
                    for i in range(len(same_s_movie)):
                        similarity, id = same_s_movie[i]
                        name = dataset.find_movie_name(id+1)
                        predict_rate = similarity * (-1) * 5
                        print("%60s || similarity = %.4f || predicted rating = %.2f" %(name, similarity * (-1), predict_rate))
                        sum += (predict_rate - um_matrix[255][id]) * (predict_rate - um_matrix[255][id])
                        n_output -= 1
                else:
                    while(n_output>0):
                        idx = random.randint(0, len(same_s_movie)-1)
                        similarity, id = same_s_movie[idx]
                        name = dataset.find_movie_name(id+1)
                        predict_rate = similarity * (-1) * 5
                        print("%60s || similarity = %.4f || predicted rating = %.2f" %(name, similarity * (-1), predict_rate))
                        sum += (predict_rate - um_matrix[255][id]) * (predict_rate - um_matrix[255][id])
                        n_output -= 1
        print(f"\n\nRMSE = {math.sqrt(sum / n_movie)}\n\n")
        if(watched_movie_name == "00"): break


if __name__ == '__main__':
    content_based()
