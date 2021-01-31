#recommendation engine for Jokes Dataset
import os
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

jokes =pd.read_excel("D:\\360Assignments\\Submission\\7 Recommendation Engine\\Ratings.xlsx")

sns.heatmap(jokes.isnull(), cmap='viridis')
# no null or NAn values found

type(jokes)
#print the average rating of every joke_id from highest to lowest
jokes.groupby('joke_id')['Rating'].mean().sort_values(ascending=False).head()

# to find  10 rating given joke_id top to bottom
jokes.groupby('joke_id')['Rating'].count().sort_values(ascending=False).head()


ratings= pd.DataFrame(jokes.groupby('joke_id')['Rating'].mean())

# number of times , it is rated 
ratings['Number of Rating'] =pd.DataFrame(jokes.groupby('joke_id')['Rating'].count())
ratings.sort_values(by= ratings['Rating'],ascending=False).head()

# to view the same data in histogram
ratings['Number of Rating'].plot.hist(bins=40)
ratings['Rating'].plot.hist(bins=90)


# creating a pivot matrix for usderid and jokeid
joke_matrix = jokes.pivot_table(index='user_id',columns='joke_id',values='Rating')
joke_matrix.head()

# picking which jokes to recommend by  Most Rated 
rating_sorted =pd.DataFrame(ratings.sort_values('Number of Rating',ascending=False))

type(joke_matrix)

#joke_id =8 and 6
joke8 = joke_matrix[8]
joke6 =joke_matrix[6]

joke8.head(10)

# corrwith 2 dataframes
similar_jokeid8= joke_matrix.corrwith(joke8)

# Now putting the dataframe of similar for joke6
similar_jokeid6= joke_matrix.corrwith(joke6)

# --- for jokeid =8------------------
# dropping Nan values to get only recommended jokes for the user
corr_joke8= pd.DataFrame(similar_jokeid8,columns=['Correlations'])
corr_joke8.dropna(inplace=True)
corr_joke8.head()

# fixing correlatng issues and miscrepancies . Only recommending the jokes 
#which users has rated more than 50 times.But first  adding (joining) the 'Number of Rating' col
corr_joke8 = corr_joke8.join(ratings['Number of Rating'])


# select jokes from  emp where Number of Rating>50 sort by Correlation desc
corr_joke8[corr_joke8['Number of Rating']>50].sort_values('Correlations', ascending=False)

#         Correlations  Number of Rating
#joke_id                                
#12           1.000000               210
#8            1.000000              1328
#14           1.000000               129
#38           1.000000               324