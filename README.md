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
>1. Content based
>2. Collaborative filtering
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
**TODO**

### RL (DQN)
**TODO**

## Experiment results
**TODO**
