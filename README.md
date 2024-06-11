# AI final project - Movie Recommendation System
112 下 陳奕廷 人工智慧概論 Group 16 final project<br>

## Introduction
Everytime when we want to watch a movie, we always spend lots of time looking for the movie to watch. And after we watch a movie, it may turns out that we don't like this movie. If there is someone who can give us some recommendation, it will be great. Therefore, we design this movie recommend system to give you some recommendations. It can give you five recommended movies base on the movie you watched. With this system, you can quickly find the movie you like.

- **For users**, they can find the movies they may enjoy, leading to a better watching experience. Also, this can save the users’ time in searching moveis they like.
- **For platform**, if they can provide the movies that match the user’s preerence, users are more likely to stay in this platform, bringing the platform a higher revenue.

## Requirement
### package
Make sure that you have installed all the required packages in requirements.txt<br>
You can use `pip install -r requirements.txt` to install the packages
```
# requirements.txt

numpy==1.26.1
torch==2.3.0
tqdm==4.66.2
pandas==2.2.2
surprise==1.1.4
gym==0.26.2
```
### dataset
We use [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dadaset. It consists of 100,000 ratings (1-5) from 943 users on 1682 movies. 
> Each user has rated at least 20 movies.<br>
> Simple demographic info for the users (age, gender, occupation, zip)<br>

## Usage
1. Type `python main.py` to start the program.
2. Choose the algorithm you want to use. 
>1. Collaborative filtering
>2. SVD
>3. DQN
>4. Exit
3. Enter your user ID (1-943) and a movie you like.
4. The program will give you five recommend movies.
5. You can keep entering the movies you like to get more recommendation, or enter 0 to exit.
## Baseline
### Content based
Calculate the similarity between movies and recommend the movies that is "similar" to the user’s movie.
> If the user watch a movie which belongs to genre A, then the system will recommend other movies that also belong to genre A.

### Collaborative (SVD)
It relies on user-item interactions. It identifies users with similar preferences and recommends items that those like-minded users have enjoyed.
> It can provide more personalized recommendations than content based approach by considering user behavior.

### RL (DQN)
A 3-layer Neural Network. It can better capture the complex relationships between users and movies, providing more accurate recommendation.

## Hyperparameters
### Collaborative (SVD)
- learning rate (lr_all)
    > When the range of learning rate become more precise, the rating would be higher.
- number of factor (n_factors)
    > When number of factors is increasing, the ratings are decreasing. 

### RL (DQN)
- learning rate (lr)
- capacity
- gamma
- batch size
- epsilon

## Experiment results
We use RMSE to evaluate the performance of content base and collaborative filtering.
### Same input
We enter the same input to both content based and collaborative filtering to get the recommendation for 1, 5, 10 times.<br>
The input comes from user 256's watched movie.
|recommend times|  content based  |  collaborative filtering  |
|:-------------:|:---------------:|:-------------------------:|
|       1       |8.261599924202647|    3.9288043553128413     |
|       5       |9.487659487650069|    3.858081586068965      |
|       10      |11.08391490881054|    3.913781415118725      |

### Input according to the recommendation of content base
We also try another input. In this senior, we enter the recommendation of content based.
|recommend times|   content based  |  collaborative filtering |
|:-------------:|:----------------:|:------------------------:|
|       1       |2.013194072167833 |   3.857669464038458      |
|       5       |1.4141995167287948|   3.7697541396635095     |
|       10      |3.217187094744785 |   3.791153271628686      |


### Which one is better?
From the result above, we can see that in different senior, different algorithm will have different RMSE. It is because content based calculate similarity according to movie genre, but users may type in different kind of movie everytime. Therefore, in senior 2, the entered movies belong to the same genre, so it has a lower RMSE and is better than collaborative filtering.

### Result of DQN
We fail to implement our DQN. We can use it to get the recommended movie, and the recommendation seems to be reasonable, but as we plot the reward, it is clearly that we have a wrong result. Due to the time limitaion, we are not able to fix this problem.
