# DiscordSocialGraph

Statistical and Machine Learning project for discovering and analysing social structures on Discord. 

## Methodology

Data is collected using [this cog](https://github.com/samclane/Snake-Cogs/blob/master/member_logger/member_logger.py) 
for [RedDiscordBot](https://github.com/Cog-Creators/Red-DiscordBot). 

The data collected is rather simple. Only 2 user actions are tracked: joining a server and mentioning users in a message.
The data is stored in a `csv` file, looking like the following:

```csv
timestamp,member,present
1538763696,user7,"['user6', 'user9', 'user2', 'user1', 'user3']"
1538764529,user4,['user5']
1538765069,user3,"['user5', 'user4', 'user6', 'user9', 'user0', 'user8', 'user1', 'user2']"
1538765338,user5,"['user2', 'user8', 'user7', 'user3', 'user4']"
1538765956,user9,"['user2', 'user1', 'user6']"
1538766375,user7,"['user0', 'user4', 'user9', 'user1', 'user6', 'user3']"
1538767133,user9,"['user7', 'user6', 'user8', 'user4', 'user2', 'user1']"
1538767950,user1,"['user3', 'user5', 'user6']"
1538768014,user0,"['user5', 'user1']"
1538768514,user5,['user4']
...
```

The `member` is the user performing the action, and `present` are either the users already in the server, or users 
mentioned in the message.

As enough real-life data hasn't been collected yet, a method to generate data has been created. `generate_samples.py` will
create dummy data to train the model on. `generate_samples.py A B` allows you to create A samples for a server with B 
members. The data generator also selects a few members to be "friends", which biases the data in order to put the 2 users
together more often. 

Sample:
```
$ python generate_samples.py 5000 10 data.csv
Created 10 members.
Generating friends...
user0 and user8 are friends with weight .58
Generating samples
Data successfully generated!
------
           member                                            present
timestamp                                                           
1538766897  user5  [user9, user6, user8, user3, user2, user4, use...
1538766964  user4                [user8, user7, user0, user5, user6]
1538767107  user1  [user4, user2, user0, user9, user3, user7, use...
1538767294  user9         [user1, user8, user5, user3, user0, user4]
1538768286  user1  [user8, user2, user4, user0, user7, user6, user5]
1538768659  user2  [user5, user1, user7, user6, user3, user9, use...
1538769032  user3  [user7, user0, user8, user2, user1, user9, use...
1538769568  user4                [user9, user7, user8, user3, user0]
1538770333  user1  [user9, user0, user3, user8, user2, user5, use...
...
[5000 rows x 2 columns]
Saving data to C:\Users\SawyerPC\PycharmProjects\DiscordSocialGraph\data.csv.
```

The data can then by processed by running `process_data.py`

```
$ python process_data.py data.csv
Encoding data...
      user0  user1  user2  user3  user4  user5  user6  user7  user8  user9
0         1      1      1      1      1      0      1      1      1      1
1         1      0      0      0      0      1      1      1      1      0
2         1      0      1      1      1      1      1      1      1      1
3         1      1      0      1      1      1      0      0      1      0
4         1      0      1      0      1      1      1      1      1      0
5         1      1      0      1      0      1      1      1      1      1
6         1      1      1      0      0      1      1      1      1      1
7         1      0      0      1      0      0      0      1      1      1
8         1      0      1      1      1      1      1      1      1      1
9         0      0      0      0      0      0      0      0      1      0
...
[5000 rows x 10 columns]
Training classifier...
Done.
Building graph...
Done. Showing graph.
```

A social graph is then constructed, with weights consisting of the Gaussian Naive Bayes probability of that user interacting with given a 
neighbor N. The graph is then drawn:

![Social Graph](https://i.imgur.com/2ItIro7.png)

A "noise floor" can be added to prune insignificant edges. A `noise_floor` of `.1` produces the following graph with the
same data:

![Pruned Graph](https://i.imgur.com/O6XGAos.png)

However, it's still not apparent that user0 and user8 are friends. This is partly due to the relatively low weight of 
.58. Weights must get into the .8-.9 range before they start making a difference. Also, due to the complete randomness 
of the data generation methods, some "strong friendships" may occur naturally, dwarfing any intentional bias. 

The main objective is to find the "linchpin" of the server; that is, the node in the graph 
with the most incoming weight, signifying they cause the server's population and interaction levels to increase the most. 
