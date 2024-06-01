import random

data = []   # string
item = []   # string
occupation = []
user = []   # string

'''
load rating (u.data)
return value: list (string)
ex: print(data[0])  # output: ['196', '242', '3', '881250949']
'''
def load_data():
    f = open('ml-100k/u.data')
    for _ in range(100000):
        line = f.readline().strip().split('\t') # user id | movie id | rating | timestamp
        data.append(line)
    f.close()
    return data

'''
load movie (u.item)
return value: list (string)
ex: print(item[0])  # output: ['1', 'Toy Story', '01-Jan-1995', '', 'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)', '0', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
'''
def load_movie():
    f = open('ml-100k/u.item', encoding='ISO-8859-1')
    for _ in range(1682):
        line = f.readline().strip().split('|')  # movie id | movie title | release date | video release date |IMDb URL | 類型(19項)
        line[1] = line[1].split(' (')[0]        # remove the movie year
        item.append(line)
    f.close()
    return item

'''
load user (u.user)
return value: list (int & string)
              gender F = 0, M = 1
              occupation = [0, 20]
ex: print(user[0])  # output: [1, 24, 1, 19, '85711']
'''
def load_user():
    g = open("ml-100k/u.occupation","r")
    for _ in range(21):
        gline = g.readline().strip()
        occupation.append(gline)
    g.close()    

    f = open("ml-100k/u.user","r")
    mp = {'T8H1N': 10000, 'V3N4P': 10001, 'L9G2B': 10002, 'E2A4H': 10003, 'V0R2M': 10004, 'Y1A6B': 10005,
          'V5A2B': 10006, 'M7A1A': 10007, 'M4J2K': 10008, 'R3T5K': 10009, 'N4T1A': 10010, 'V0R2H': 10011,
          'K7L5J': 10012, 'V1G4L': 10013, 'L1V3W': 10014, 'N2L5N': 10015, 'E2E3R': 10016}

    for _ in range(943):
            line = f.readline().strip().split('|') # user id | age | gender | occupation | zip code
            line[0] = int(line[0])
            line[1] = int(line[1])
            if(line[2] == 'F'): line[2]=0
            elif(line[2] == 'M'): line[2]=1

            if(line[4].isdigit()): line[4] = int(line[4])
            else:
                line[4] = mp[line[4]]
            
            for i in range(21):
                if(line[3]==occupation[i]): 
                    line[3] = i
                    break
                else: continue
            user.append(line)
    f.close()
    return user

# given movie name, find movie id (int) (array index, so return id-1)
def find_movie_id(movie_name):
    for i, sublist in enumerate(item):
        if movie_name in sublist:
            movie_id = sublist[0]
            return int(movie_id)-1
    # not found
    print(f"In find_movie_name in dataset.py, the movie: {movie_name} is not in the dataset, please change a movie.")
    return -1

# given movie id (int), find movie name
def find_movie_name(movie_id):
    if(type(movie_id) != int):
        print(f"In find_movie_name() in dataset.py, movie_id should be int, but receive {type(movie_id)}")
        raise ValueError()
    for i, sublist in enumerate(item):
        if str(movie_id) in sublist:
            movie_name = sublist[1]
            return movie_name
    # not found
    print(f"In find_movie_name in dataset.py, the movie id: {movie_id} is not in the dataset, please change a movie.")
    return ""

def find_next_movie(user_id, movie_id):
    temp = []
    like = []
    time = 0
    for i in range(100000):
        if(user_id == int(data[i][0])): temp.append(data[i])
        if(user_id == int(data[i][0]) and movie_id == int(data[i][1])): time = int(data[i][3])
    
    nexttime = 899999999
    flag = 0
    count = 0
    for index,array in enumerate(temp):
        if(int(array[2])>=3):
            like.append(array)
            count = count + 1
        if(int(array[3])>time and int(array[3])<nexttime and int(array[2])>= 3): 
            flag = 1
            nexttime = int(array[3])
            nextindex = index

    if(flag == 0): 
        return int(like[random.randint(0,count-1)][1])
    
    return int(temp[nextindex][1])
