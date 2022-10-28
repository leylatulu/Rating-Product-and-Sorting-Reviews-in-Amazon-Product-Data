# let's load required libraries
import pandas as pd
import math
import scipy.stats as st

# some settings
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

######################################################################################################
######################################################################################################
# examine the variables

# reviewerID : User Id
# asin : Product Id
# reviewerName : User name
# helpful : Useful evaluation rating
# reviewText : Evaluation
# overall : Product rating
# summary : Evaluation summary
# unixReviewTime : Evaluation time
# reviewTime : Evaluation time (RAW)
# day_diff : Number of days since evaluation
# helpful_yes : The number of times the evaluation was found useful
# total_vote : Number of votes given to the evaluation
######################################################################################################
######################################################################################################
# load data and create copy
df_ = pd.read_csv("amazon_review.csv")
df = df_.copy()

# overview
df.shape # (4915, 12)
df.describe().T

"""
                    count             mean            std              min              25%              50%              75%              max
overall        4915.00000          4.58759        0.99685          1.00000          5.00000          5.00000          5.00000          5.00000
unixReviewTime 4915.00000 1379465001.66836 15818574.32275 1339200000.00000 1365897600.00000 1381276800.00000 1392163200.00000 1406073600.00000
day_diff       4915.00000        437.36704      209.43987          1.00000        281.00000        431.00000        601.00000       1064.00000
helpful_yes    4915.00000          1.31109       41.61916          0.00000          0.00000          0.00000          0.00000       1952.00000
total_vote     4915.00000          1.52146       44.12309          0.00000          0.00000          0.00000          0.00000       2020.00000
"""

df.columns

"""
(['reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText','overall', 'summary', 
'unixReviewTime', 'reviewTime', 'day_diff', 'helpful_yes', 'total_vote'], dtype='object')
"""

######################################################################################################
######################################################################################################
# In the dataset, users rated a product and made comments. We will weight the scores given by date.

# Calculate average score for product
df.overall.mean() # 4.587589013224822

# Calculate weighted average score by date
df.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4915 entries, 0 to 4914
Data columns (total 12 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   reviewerID      4915 non-null   object 
 1   asin            4915 non-null   object 
 2   reviewerName    4914 non-null   object 
 3   helpful         4915 non-null   object 
 4   reviewText      4914 non-null   object 
 5   overall         4915 non-null   float64
 6   summary         4915 non-null   object 
 7   unixReviewTime  4915 non-null   int64  
 8   reviewTime      4915 non-null   object 
 9   day_diff        4915 non-null   int64  
 10  helpful_yes     4915 non-null   int64  
 11  total_vote      4915 non-null   int64  
dtypes: float64(1), int64(4), object(7)
memory usage: 460.9+ KB
"""

# convert reviewTime to datetime
df.reviewTime = pd.to_datetime(df.reviewTime)

# current_date is considered as max value of reviewTime
current_date = df.reviewTime.max()


# take the difference between each review date and current_date in days and create a new variable as days
df["days"] = (current_date-df.reviewTime).dt.days

# use the quantile function to divide the variable into 4 parts (if 3 quartiles are given, 4 pieces come out)
q1 = df["days"].quantile(0.25) # 280.0
q2 = df["days"].quantile(0.50) # 430.0
q3 = df["days"].quantile(0.75) # 600.0

# define a function to calculate the weighted average score (weights randomly selected
def user_based_weighted_average(dataframe, w1=40, w2=30, w3=20, w4=10):
    a = dataframe.loc[dataframe["days"] <= 280, "overall"].mean() * w1 / 100
    b = dataframe.loc[(dataframe["days"] > 280) & (dataframe["days"] <= 430), "overall"].mean() * w2 / 100
    c = dataframe.loc[(dataframe["days"] > 430) & (dataframe["days"] <= 600), "overall"].mean() * w3 / 100
    d = dataframe.loc[dataframe["days"] > 600, "overall"].mean() * w4 / 100
    return a + b + c + d, a, b, c, d

# Compare the average of each time interval in the weighted rating
total, a, b, c, d = user_based_weighted_average(df) # 4.628116998159475

"""
total = 4.628116998159475
a = 1.8783171521035598
b = 1.3908421913327882
c = 0.9143322475570033
d = 0.44462540716612375
"""
# When weighting the scores, more weight is given to the most recent ratings.
# For this reason, higher score are obtained in new ratings.

# Calculate average score without weighting by date
df.loc[df["days"] <= 280, "overall"].mean() # 4.6957928802588995
df.loc[(df["days"] > 280) & (df["days"] <= 430), "overall"].mean() # 4.636140637775961
df.loc[(df["days"] > 430) & (df["days"] <= 600), "overall"].mean() # 4.571661237785016
df.loc[df["days"] > 600, "overall"].mean() # 4.4462540716612375

######################################################################################################
######################################################################################################

# total_vote means the total number of up-down votes given to a comment.
# up means helpful
# Find the number of unhelpful votes
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# Define a function to calculate the score_pos_neg_diff
def score_pos_neg_diff(up, down):
    return up-down

# Define a function to calculate the score_average_rating
def score_average_rating(up, down):
    # Zero division hatası için
    if up + down == 0:
        return 0
    return up/(up+down)

# Define a function to calculate the wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound score
    - The lower bound of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used to for product ranking.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float


    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# Calculate the score_pos_neg_diff, score_average_rating, wilson_lower_bound score and save it in the df
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]),axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


# Select the first 20 comments according to wilson_lower_bound
df.sort_values("wilson_lower_bound", ascending= False).head(20)

# While determining the 20 comments, they were sorted according to the wilson_lower_bound score.
# Negative reviews also matter as social proof is taken into account when wlb ranking.















