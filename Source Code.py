# %%
import encodings
import snscrape.modules.twitter as smt
import pandas as pd
import numpy as np

#Scraping Covid19 tweets made in January 2022
filename1 ='Jan_tweets'
queryJan = 'Covid19 Vaccine since:2022-01-01 until:2022-01-30'
tweets_Jan=[]
limit =200

for tweet in smt.TwitterSearchScraper(queryJan).get_items():

   if len(tweets_Jan) == limit:
      break
   else:
      tweets_Jan.append([ tweet.content])
    
df =pd.DataFrame(tweets_Jan, columns=['content'])
print(df)
#Storing January tweets in Jan_tweets file
df.to_json(f'{filename1}.JSON')


#Scraping Covid19 tweets made in June 2020
queryJune = 'Covid19 Vaccine since:2020-05-25 until:2020-06-30'
tweets_June =[]
limit = 200
filename2 ='June_tweets'
for tweet in smt.TwitterSearchScraper(queryJune).get_items():

   if len(tweets_June) == limit:
      break
   else:
      tweets_June.append([ tweet.content])
    
dfJune =pd.DataFrame(tweets_June, columns=['content'])
print(dfJune)
#Storing June tweets in June_tweets file
dfJune.to_json(f'{filename2}.JSON')

# %%
#Cleaning the data

import pandas as pd
#Identifying null values
def inds_nans(df):
    inds = df.isna().any(axis=1)
    return inds
#Identifying duplicates
def inds_dups(df):
    inds = df.duplicated()
    return inds

lstcontent =[]
l1=[]
l2=[]
l3=[]

df1 = pd.read_json('Jan_tweets.JSON')
data_clean = df1.loc[~((inds_nans(df1) | inds_dups(df1) )),:]
data_clean

print(data_clean)
content = data_clean['content'].to_string()
lstcontent = content.split()
for i in lstcontent:
    if i.isalnum() and not i.isdigit():
        l1.append(i)
for i in range(len(l1)):
    if i%2==0:
      l2.append(l1[i])
    else:
      l3.append(l1[i])

data_clean.insert(1,'contentfrom',l2[:200])
data_clean.insert(2,'contentto',l3[:200])
data_clean

# %%
# Constructing and Plotting the word co-occurrence network
import networkx as nx
import pandas as pd

graph_Jan = nx.from_pandas_edgelist(data_clean, source ="contentfrom", target ="contentto", create_using =nx.Graph())
type(graph_Jan)
print(nx.info(graph_Jan))

graph_Jan.nodes()
len(graph_Jan.nodes())
graph_Jan.edges()
len(graph_Jan.edges())
print('edges',graph_Jan.edges())
nx.draw(graph_Jan)

# ploting the graph
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
pos = nx.random_layout(graph_Jan)
nx.draw_networkx(graph_Jan,with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues,pos=pos)
plt.show()


# %%
import networkx as nx

nd_view = graph_Jan.degree()
print(nd_view)
Jan_nodes = list(graph_Jan.nodes())
Jan_deg = list(dict(nd_view).values())
print(Jan_deg, Jan_nodes)


# %%
#Creating a dataframe of nodes and degrees and computing degree distribution
import pandas as pd
first = 'first'
Jandata = {'Node':Jan_nodes, 'degree':Jan_deg}
dfJan = pd.DataFrame(Jandata)
print(dfJan)


# %%
#Network measures
# Degree distribution
plt.hist([v for k,v in nx.degree(graph_Jan)])
plt.xlabel('Degree')
plt.ylabel('Node count')
plt.title('Degree distribution of January,2022 tweets')
plt.show()
print(nx.degree(graph_Jan))

# %%
#Betweenness Centrality
Betweenness_centrality = nx.centrality.betweenness_centrality(graph_Jan).values()
print(Betweenness_centrality)
print('Maximum Betweenness centrality - Jan:',max(Betweenness_centrality))
plt.hist(Betweenness_centrality)
plt.xlabel('Betweenness Centrality score')
plt.ylabel('Frequency')
plt.title('Betweenness Centrality of January,2022 tweets')
plt.show()

# %%
#Closeness centrality
closeness_centrality = nx.centrality.closeness_centrality(graph_Jan).values() 
print(closeness_centrality)
print('Maximum closeness centrality - Jan:',max(closeness_centrality))
plt.hist(closeness_centrality)
plt.xlabel('Closeness Centrality score')
plt.ylabel('Frequency')
plt.title('Closeness Centrality of January,2022 tweets')
plt.show()

# %%
#Cleaning the data

import pandas as pd

#Identifying missing values
def inds_nans(df):
    inds = df.isna().any(axis=1)
    return inds

#Identifying duplicates
def inds_dups(df):
    inds = df.duplicated()
    return inds

lstcontent =[]
l1=[]
l2=[]
l3=[]

df1 = pd.read_json('June_tweets.JSON')
data_clean_June = df1.loc[~((inds_nans(df1) | inds_dups(df1) )),:]
data_clean_June


print(data_clean_June.head())
content = data_clean_June['content'].to_string()
lstcontent = content.split()
for i in lstcontent:
    if i.isalnum() and not i.isdigit():
        l1.append(i)


for i in range(len(l1)):
    if i%2==0:
      l2.append(l1[i])
    else:
      l3.append(l1[i])

data_clean_June.insert(1,'contentfrom',l2[:199])
data_clean_June.insert(2,'contentto',l3[:199])
data_clean_June.head()

# %%
# Constructing and Plotting the word co-occurrence network
import networkx as nx
import pandas as pd
# plot the graph
graph_June = nx.from_pandas_edgelist(data_clean_June, source ="contentfrom", target ="contentto", create_using =nx.Graph())
type(graph_June)
print(nx.info(graph_June))

graph_June.nodes()
len(graph_June.nodes())
graph_June.edges()
len(graph_June.edges())
print('edges',graph_June.edges())
nx.draw(graph_June)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
pos = nx.random_layout(graph_June)
nx.draw_networkx(graph_June,with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues,pos=pos)
plt.show()


# %%
import networkx as nx

nd_view = graph_June.degree()
print(nd_view)
June_nodes = list(graph_June.nodes())
June_deg = list(dict(nd_view).values())
print(June_deg, June_nodes)


# %%
#Creating a dataframe of nodes and degrees 
import pandas as pd
first = 'first'
Junedata = {'Node':June_nodes, 'degree':June_deg}
dfJan = pd.DataFrame(Junedata)
print(dfJan)
dfJan.to_json(f'{first}.JSON')

print(dfJan.degree.size)


# %%
# Degree distribution
plt.hist([v for k,v in nx.degree(graph_June)])
plt.xlabel('Degree')
plt.ylabel('Node count')
plt.title('Degree distribution of June,2020 tweets')
plt.show()
print(nx.degree(graph_June))

# %%
Betweenness_centrality_June = nx.centrality.betweenness_centrality(graph_June).values()
print(Betweenness_centrality_June)
plt.hist(Betweenness_centrality_June)
print('Maximum betweenness centrality - June:',max(Betweenness_centrality_June))
plt.xlabel('Betweenness Centrality score')
plt.ylabel('Frequency')
plt.title('Betweenness Centrality of June,2020 tweets')
plt.show()

# %%
#Closeness centrality
closeness_centrality = nx.centrality.closeness_centrality(graph_June).values() 
print(closeness_centrality)
plt.hist(closeness_centrality)
print('Maximun Closeness Centrality - June:',max(closeness_centrality))
plt.xlabel('Closeness Centrality score')
plt.ylabel('Frequency')
plt.title('Closeness Centrality of June,2020 tweets')
plt.show()

