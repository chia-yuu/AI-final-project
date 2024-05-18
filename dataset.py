data = []   # string
item = []   # string

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
