
# Building a Recommendation System with Collaborative Filtering (ALS)
---
# Stage 1: Data Collection and Pre-processing


## Objectives:
* Demonsrate an understanding on how recommendation systems are being used for personalization of online services/products.
* Parse and filter datasets into Spark RDDs, performing basic feature selection. 
* Run a hyper-parameter selction activity through a scalable grid search. 
* Train and evaluate the predictive performance of recommendation system.
* Generate predictions from the trained model.



![](https://cdn-images-1.medium.com/max/800/1*Zvwzw_KPRv5bcXPkb6WubA.jpeg)

## Introduction

Recommender/Recommendation Systems are one of the most successful applications of machine learning in the Big Data domain. Such systems are integral parts in the success of Amazon (Books, Items), Pandora/Spotify (Music), Google (News, Search), YouTube (Videos) etc.  For Amazon these systems bring more than 30% of their total revenues. For Netflix service, 75% of movies that people watch are based on some sort of recommendation.

> The goal of Recommendation Systems is to find what is likely to be of interest to the user. This enables organizations to offer a high level of personalization and customer tailored services.



For online video content services like Netflix and Hulu, the need to build robust movie recommendation systems is extremely important. An example of recommendation system is such as this:

    User A watches Game of Thrones and Breaking Bad.
    User B performs a search query for Game of Thrones.
    The system suggests Breaking Bad to user B from data collected about user A.
    


This lab will guide you through a step-by-step process into developing such a movie recommendation system. We shall use the MovieLens dataset to build a movie recommendation system using collaborative filtering technique with Spark's Alternating Least Saqures implementation.

## MovieLens Ratings Dataset

Social computing research centre at university of Minnesota, [GroupLens Research](https://grouplens.org/),  has developed a movie ratings dataset called the [MovieLens](http://movielens.org/). The datasets were collected over various periods of time and can be directly downloaded from [this location](http://grouplens.org/datasets/movielens/). 

A data dictionary with details on individual datasets and included features can be viewed [HERE](http://files.grouplens.org/datasets/movielens/ml-20m-README.html)

For our experiment , we shall download the latest datasets direct from the website in the zip format. Grouplens offer the complete ratings dataset and a small subset for experimentation. We shall down both these datasets for building our recommendation system. 

* **Small Dataset**: 100,000 ratings applied to 9,000 movies by 700 users. Last updated 10/2016.

* **Complete Dataset**: 26,000,000 ratings applied to 45,000 movies by 270,000 users. Last updated 8/2017.


For this lab, we will use the small dataset `ms-latest-small.zip` which can be downloaded from the above location. The main reason for using this dataset is to speed up computation and focus more on the pyspark programming. 

* Create a folder `datasets` on your machine at a location which is accessible by pyspark. 
* Unzip the contents of downloaded zip file into `datasets` directory. 
* You may also download the complete dataset `ml-latest.zip` and save it at this location for later experimentation.

Above actions will generate following file and directory structure:

![](path.png)

Let's also import PySpark to our Python environment and and initiate a local SparkContext `sc`.


```python
import pyspark
sc = pyspark.SparkContext('local[*]') # [*] represents a local context i.e. no cluster
```

# Stage 1: Dataset Parsing, Selection and Filtering

With our SparkContext initialized, and our dataset in an accessible locations, we can now parse the csv files and read them into RDDs as shown in the previous labs. The small dataset contains a number of csv file with features as shown below:  

> #### ratings.csv :**UserID, MovieID, Rating, Timestamp**

> #### movies.csv :**MovieID, Title, Genres > *Genre1|Genre2|Genre3...**

> #### tags.csv :**UserID, MovieID, Tag, Timestamp**

> #### links.csv :**MovieID, ImdbID, TmdbID**

The complete dataset contains some other files as well. We shall focus on `ratings.csv`, and `movies.csv` from small dataset here for building a basic recommendation system. Other features can be incorporated later for improving the predictive performance of the system.  The format of these files is uniform and simple and such comma delimited files can be easily parsed line by line using Python `split()` once they are loaded as RDDs. 

We shall first parse `ratings.csv` and `movies.csv` files into two RDDs. We also need to filter out the header row in each file containing column names. 

> **For each line in the `ratings.csv`, create a tuple of (UserID, MovieID, Rating). Drop the Timestamp feature. **

> **For each line in the `movies.csv`, Create a tuple of (MovieID, Title). Drop the Genres. **

Set path variables for `ratings` and `movies` files. 


```python
# Create a path for identifying the ratings and movies files in small dataset
ratingsPath = 'datasets/ml-latest-small/ratings.csv'
moviesPath = 'datasets/ml-latest-small/movies.csv'
```

### Parsing `ratings.csv`

Read the contents of ratings file into an RDD and view its first row as header


```python
# Use .textFile() to read the raw contents of ratings file into an RDD
# read the first line of this RDD as a header and view header contents

ratingsRaw = sc.textFile(ratingsPath)
ratingsHeader = ratingsRaw.take(1)[0]
ratingsHeader

# 'userId,movieId,rating,timestamp'
```




    'userId,movieId,rating,timestamp'



We need to filter some of the data at this stage. We can drop the timestamp feature, parse the raw data into a new RDD and filter out the header row. Perform following transformations on `ratingsRaw`:

1. Read `ratingsRaw` contents into a new RDD while using `.filter()` to exclude the header information.
2. Split each line of the csv file using `,` as the input argument with `split()` function.
3. Collect the first three elements of each row (UserID, MovieID, Rating) and discard timestep field.
4. Cache the final RDD (Optional) using `RDD.cache()` (may help speed up computation with large RDDs).
5. Print the total number of recommendations and view first three rows.


```python
ratingsNoHeader= ratingsRaw.filter(lambda line: line != ratingsHeader )
ratingsSplit = ratingsNoHeader.map(lambda line: line.split(","))
ratingsRDD = ratingsSplit.map(lambda x: (int(x[0]),int(x[1]),float(x[2])))
ratingsRDD.cache()

print ("There are %s recommendations in the  dataset" % (ratingsRDD.count()))

ratingsRDD.take(3)

# There are 100004 recommendations in the  dataset
# [(1, 31, 2.5), (1, 1029, 3.0), (1, 1061, 3.0)]
```

    There are 100004 recommendations in the  dataset
    




    [(1, 31, 2.5), (1, 1029, 3.0), (1, 1061, 3.0)]



This looks well in-line with our expectations. Let's do the same for `movies.csv`.

### Parsing `movies.csv`

We shall now proceed in a similar way with `movies.csv` file. Repeat following steps as performed above:

1. Use the path variable for identifying the location of **movies.csv**.
2. Read the text file into RDD.
3. Exclude the header information.
4. Split the line contents of the csv file.
5. Read the contents of resulting RDD creating a (MovieID, Title) tuple and discard genres. 
6. Count number of movies in the final RDD.


```python
moviesRaw = sc.textFile(moviesPath)
moviesHeader = moviesRaw.take(1)[0]
moviesRDD = moviesRaw.filter(lambda line: line != moviesHeader)\
                          .map(lambda line: line.split(","))\
                          .map(lambda x: (x[0], x[1]))\
                          .cache()
print ("There are %s movies in the complete dataset" % (moviesRDD.count()))

# There are 9125 movies in the complete dataset
```

    There are 9125 movies in the complete dataset
    

We now have the two RDDs we created above and we shall use these to build and train our recommendation system. 

### Saving Pre-Processed Data (optional)

We can optionally save our preprocessed datasets. Create a folder "processed" and save `movieRDD` and `ratingsRDD` using `RDD.saveAsTExtFile(filename)`. 


```python
# Create a directory "processed" and store the preprocessed dataset RDDs as text files using .saveAsTExtFiles() method. 
import os 

processedPath = 'processed'
os.mkdir(processedPath)

moviesRDD.saveAsTextFile(os.path.join(processedPath, 'moviesRDD'))
ratingsRDD.saveAsTextFile(os.path.join(processedPath, 'ratingsRDD'))
```

---
# Stage 2: Alternate Least Squares: Model Training and Evaluation

### Splitting the Data as Testing , Training and Validation Sets. 

We can now go ahead and split the data for building our recommendation system. We can use spark's `randomsplit()` transformation that uses given weights to split an rdd into any number of sub-RDDs. The standared usage of this transformation function is :

> `RDD.randomSplit(weights, seed)`

**weights** – weights for splits, will be normalized if they don’t sum to 1

**seed** – random seed

Let's split the `ratingsRDD` into training, testing and validation RDDs (60%, 20%, 20%) using respective weights.
Perform a `.count` on resulting datasets to view the count of elements of each RDD. 


```python
# Split ratingsRDD into training, validation and testing RDDs as 60/20/20
# Set seed to 100 for reproducibility
# Show the count of each RDD

trainRDD, validRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=200)

trainRDD.count(), testRDD.count(), validRDD.count()

# (60050, 19904, 20050)
```




    (59702, 20142, 20160)



For prediction of ratings, we would need `customerID` and `movieID` from validation and test RDDs respectively. Let's map these values into two new RDDs which will be used for training and validation purpose. We shall ignore the `ratings` values for these RDDs, as these will be predicted later.  Take 3 elements from both RDDs to inspect the results.


```python
# Read customer ID and movie ID from validation and test sets. DO NOT show ground truth (ratings) to the model 

validFeaturesRDD = validRDD.map(lambda x: (x[0], x[1]))
testFeaturesRDD = testRDD.map(lambda x: (x[0], x[1]))

print ('Validation Features:', validFeaturesRDD.take(3))
print ('Test Features:', testFeaturesRDD.take(3))

# Validation Features: [(1, 1263), (1, 1343), (1, 1405)]
# Test Features: [(1, 1129), (1, 2294), (1, 2968)]
```

    Validation Features: [(1, 1263), (1, 1343), (1, 1405)]
    Test Features: [(1, 1129), (1, 2294), (1, 2968)]
    

We will use the `validFeaturesRDD` during the training process to avoid the model from overfitting / getting stuck into a local minima and the `testFeaturesRDD` with trained model to measure its predictive performance. 

## Collaborative Filtering

Collaborative filtering allows us to make predictions **(filtering)** about the interests of a user by collecting preferences or taste information from many users **(collaborating)**. 

The key idea is that if a user A has the same opinion as a user B on an issue/object, A is more likely to have a similar opinion as user B on a different issue, than to have s opinion similar to that of a user chosen randomly. 

Following image shows an example of collaborative filtering. Initially, people rate different items (songs, videos, images, games), followed by the system making predictions about a user's rating for an item that he has not not rated yet. The new predictions are built upon the existing ratings of other users with similar ratings with the active user. 
![](https://slideplayer.com/slide/5692490/18/images/7/Collaborative+Filtering.jpg)


Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares (ALS) algorithm.

## Alternate Least Squares in Spark

Collaborative filtering is commonly used for recommender systems. This algorithm aims to fill in the missing entries of a user-item association matrix as shown in the figure above. `spark.mllib` currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. `spark.mllib` uses the alternating least squares (ALS) algorithm to learn these latent factors. 


We shall work with following hyper-parameters and set their values prior to the actual training process:

* `rank` : Number of hidden/latent factors in the model. **(use the list [2,4,6,8,10] as rank values)**
* `iterations` : Number of iterations to run. **(initially set to 10)**
* `lambda` :  Regularization parameter in ALS.**(set to 0.1)**

Spark offers a lot of other parameters for detailed and indepth fine tuning of the algorithm. Details on spark's ALS implementation can be viewed [HERE](https://spark.apache.org/docs/2.2.0/mllib-collaborative-filtering.html). For now, we will use default values for all the other hyper parameters. 

Let's import the ALS algorithm rom spark's machine learning library `mllib` and set parameters shown above. For this experiment, we shall use `iterations = 10`, `lambda = 0.1` and run a grid for identifying best value for `rank`. We need to import ALS from `mllib` along with `math` module (for calculating `RMSE`) and set our learning parameters. 

> **Note**: You may decide to run a larger grid with other model parameters after setting up the codebase.


```python
# Import ALS from spark's mllib
from pyspark.mllib.recommendation import ALS
import math

# set learning parameters 
seed = 500
numIterations = 10
lambdaRegParam = 0.1
ranksVal = [2, 4, 6, 8, 10]
errVal = [0, 0, 0, 0, 0] # initialize a matrix for storing error values for selected ranks
errIter = 0 # iterator for above list 

# Set training parameters
minError = float('inf')
bestRank = -1 
bestIteration = -1
```

### Model Training and Validation for hyper-parameter optimization

We can now start our training process using above parameter values which would include following steps: 

* Run the training for each of the rank values in our `ranks` list inside a for loop.
* Train the model using trainRDD, ranksVal, seed, numIterations and lambdaRegParam value as model parameters. 
* Validate the trained model by predicting ratings for `validFeaturesRDD` using `ALS.predictAll()`.
* Compare predicted ratings to actual ratings by joining generated predictions with `validRDD`. 
* Calculate error as RMSE for each rank. 
* Find the best rank value based on RMSE

For sake of simplicity, we shall repeat training process for changing ranks value **only**. Other values can also be changed as a detailed grid search for improved predictive performance. 


```python
# Run ALS for all values in ranks
for r in ranksVal:
    
    # Train the model using trainRDD, rank, seed, iterations and lambda value as model parameters
    movieRecModel = ALS.train(trainRDD, 
                              rank = r, 
                              seed = seed, 
                              iterations = numIterations,
                              lambda_ = lambdaRegParam)
    
    # Use the trained model to predict the ratings from validPredictionRDD using model.predictAll()
    predictions = movieRecModel.predictAll(validFeaturesRDD).map(lambda p: ((p[0], p[1]), p[2]))

    # Compare predicted ratings and actual ratings in validRDD
    validPlusPreds = validRDD.map(lambda p: ((int(p[0]), int(p[1])), float(p[2]))).join(predictions)
    
    # Calculate RMSE error for the difference between ratings and predictions
    error = math.sqrt(validPlusPreds.map(lambda p: (p[1][0] - p[1][1])**2).mean())
    
    # save error into errors array
    errVal[errIter] = error
    errIter += 1
    
    print ('For Rank = %s , the RMSE value is: %s' % (r, error))
    
    # Check for best error and rank values
    if error < minError:
        minError = error
        bestRank = r

print ('The best model was trained with Rank = %s' % bestRank)

# For the selected rank: 2 , the RMSE is: 0.9492876773915179
# For the selected rank: 4 , the RMSE is: 0.94553209880163
# For the selected rank: 6 , the RMSE is: 0.9491943433112304
# For the selected rank: 8 , the RMSE is: 0.9512400007129131
# For the selected rank: 10 , the RMSE is: 0.9563593454968813
# The best model was trained with rank value = 4
```

    For Rank = 2 , the RMSE value is: 0.9330471538259933
    For Rank = 4 , the RMSE value is: 0.9451598669252498
    For Rank = 6 , the RMSE value is: 0.9486233108697174
    For Rank = 8 , the RMSE value is: 0.9565547745489039
    For Rank = 10 , the RMSE value is: 0.9537457932314284
    The best model was trained with Rank = 2
    

### Analyzing the Predictions

Let's have a look at the format of predictions the model generated during last validation stage. 


```python
# take 3 elements from the predictions RDD
predictions.take(3)

# [((580, 1084), 3.492776257690668),
#  ((302, 1084), 3.078629750519478),
#  ((514, 1084), 3.985426769882686)]
```




    [((303, 5618), 4.10284449297734),
     ((443, 5618), 5.066852128567025),
     ((429, 5618), 2.41945480252752)]



The output shows we have the `((UserID,  MovieID), Rating)` tuple, similar to the ratings dataset. The `Ratings` field in the predictions RDD refers to the ratings predicted by the trained ALS model. 

Then we join these predictions with our validation data and the result looks as follows:


```python
# take 3 elements from the validPlusPreds
validPlusPreds.take(3)

# [((1, 1405), (1.0, 2.7839431097640492)),
#  ((2, 296), (4.0, 3.9729953606585244)),
#  ((2, 616), (3.0, 3.128218990007167))]
```




    [((1, 1263), (2.0, 2.537866857914612)),
     ((1, 1343), (2.0, 3.0539445791129167)),
     ((1, 1405), (1.0, 2.7531149594637028))]



This output shows the format `((UserId, MovieId), Ratings, PredictedRatings)`. 

We then calculated the RMSE by taking the squred difference and calculating the mean value as our `error` value.

### Testing the Model

We shall now test the model with test dataset hich has been kept away from the learning phase upto this point. 
Use following parameters:
* Use `trainRDD` for training the model.
* Use `bestRank` value learnt during the validation phase.
* Use other parameter values same as above. 
* Generate predictions with `testFeaturesRDD`
* Calculate error between predicted values and ground truth as above.


```python
# Train and test the model with selected parameter bestRank

movieRecModel = ALS.train(trainRDD, 
                           bestRank, 
                           seed = seed, 
                           iterations = numIterations,
                           lambda_ = lambdaRegParam)

# Calculate predictions for testPredictionRDD
predictions = movieRecModel.predictAll(testFeaturesRDD).map(lambda x: ((x[0], x[1]), x[2]))

# Combine real ratings and predictions
testPlusPreds = testRDD.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)

# Calculate RMSE
error = math.sqrt(testPlusPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
    
print ('For testing dataset, the calculated RMSE value:  %s' % (error))

# For testing data the RMSE is 0.9498348141480232
```

    For testing dataset, the calculated RMSE value:  0.9289581787940366
    

Due to probablistic nature of ALS algorithm, changing the seed value will also show somen fluctuations in RMSE. 


---
# Stage 3: Making Recommendations

With collaborative filtering, generating new recommendations is not as straightforward as predicting new entries using a previously generated model as shown above. For collaborative filtering, we have to re-train the model including the new user preferences in order to compare them with other users in the dataset. In simple terms, the system needs to be trained every time we have new user ratings. 

Once we have our model trained, we can reuse it to obtain top recomendations for a given user or an individual rating for a particular movie. 

First we need to count the number of ratings per movie. We can create a function that inputs the movies RDD created earlier  and calculates total number of ratings. Based on this, we can later define a threshold ratings value to only include movies with a minimum count of ratings. 

Create a function `getRatingCount()` to do following:

* Input the ratings RDD (grouped by movies)
* Count the total number of rating for a given movie ID
* Return the movie id and total number of ratings as a tuple. 


Perform following tasks in the given sequence: 

* Use `ratingsRDD` to get movie ID and ratings values, and `groupby()` movie ID to group all ratings for each movie
* Pass the new RDD to the function above to count the number of ratings for all movies
* create a new RDD `movieRatingsCountsRDD` to carry movie rating and count as a tuple
* take 5 elements for inspection



```python
def getRatingCount(IDRatings):
    nRatings = len(IDRatings[1])
    return IDRatings[0], nRatings

movieRatingsCountsRDD = ratingsRDD.map(lambda x: (x[1], x[2])).groupByKey()\
                                 .map(getRatingCount)
movieRatingsCountsRDD.take(5)

# [(1172, 46), (2150, 36), (2294, 53), (2968, 43), (10, 122)]
```




    [(31, 42), (1029, 42), (1061, 33), (1129, 48), (1172, 46)]



### Adding New User(s) and Rating(s)

In order to make recommendations, we now need to create a new user and generate some initial set of ratings for collaborative filtering to work. First let's create a new user with a unique id , say 0, as its not currently used and would be easily identifiable later. 




```python
newUserID = 0
```

Now we need to rate some movies under this user. You are encouraged to look into movies RDD to set some ratings for the movies based on your own preferences. That would give you a good chance to qualitatively assess the the outcome for this system. 

For this experiment, lets create some rating values for our new user who likes comedy, family and romantic movies. You can add or omit other ratings too. 

    18	    Four Rooms (1995)	Comedy
    60074	Hancock (2008)	Action|Adventure|Comedy|Crime|Fantasy
    19	    Ace Ventura: When Nature Calls (1995)	Comedy
    203	    To Wong Foo, Thanks for Everything! Julie Newmar (1995)	Comedy
    205	    Unstrung Heroes (1995)	Comedy|Drama
    8784	Garden State (2004)	Comedy|Drama|Romance
    55830	Be Kind Rewind (2008)	Comedy
    56176	Alvin and the Chipmunks (2007)	Children|Comedy
    63393	Camp Rock (2008)	Comedy|Musical|Romance
    64622	Reader, The (2008)	Drama|Romance
    65088	Bedtime Stories (2008)	Adventure|Children|Comedy
    78499	Toy Story 3 (2010)	Adventure|Animation|Children|Comedy|Fantasy|IMAX

We will put these ratings in a new RDD use the user ID = -1 to create a (userID, movieID, rating) tuple.



```python
# Based on above, create an RDD containing (userID, movieID, rating) tuple
newUserRating = [(0,18,4),(0,60074,5),(0,19,4),(0,203,3),(0,205,4),(0,8784,5),(0,55830,3),(0,63393,4),(0,64622,5) ,(0,78499,5)]

newUserRDD = sc.parallelize(newUserRating)
```

Let's quickly check the contents of the newUserRDD to see if it meets our expectations.


```python
newUserRDD.take(10)
```




    [(0, 18, 4),
     (0, 60074, 5),
     (0, 19, 4),
     (0, 203, 3),
     (0, 205, 4),
     (0, 8784, 5),
     (0, 55830, 3),
     (0, 63393, 4),
     (0, 64622, 5),
     (0, 78499, 5)]



This looks great. We can now combine the `newUserRDD` with `moviesRDD` using a `.union()` transformation to make it a part of MovieLense dataset. Its always a good idea to check the results with `.take()`. 


```python
dataWithNewUser = ratingsRDD.union(newUserRDD)
dataWithNewUser.take(3)

[(1, 31, 2.5), (1, 1029, 3.0), (1, 1061, 3.0)]
```




    [(1, 31, 2.5), (1, 1029, 3.0), (1, 1061, 3.0)]



Now we can train the ALS model again, using all the parameters we selected before).


```python
# Train the model with `dataWithNewRating` and parameters used earlier.
newRatingsModel  = ALS.train(dataWithNewUser, 
                              rank = bestRank, 
                              seed=seed, 
                              iterations=numIterations, 
                              lambda_= lambdaRegParam)

```

We shall need to repeat that every time a user adds new ratings. Ideally we will do this in batches, and not for every single rating that comes into the system for every user.

### Getting Top Recomendations

After traning the model with our new user and ratings, we can finally get some recommendations. For that we will make an RDD with all the movies the new user hasn't rated yet.

For this stage, perform following transformations:
* Create a `moviesTitles` RDD with tuples as (id, title) from `moviesRDD`. Confirm the output.
* Make an RDD to just the IDs of the movies rated by the new user above in `newUserRating`. 
* Filter the `moviesRDD` into a new RDD `newUserUnratedRDD` to only contain those movies not rated by the user.
* Use `newUserUnratedRDD` and predict the ratings.


```python
# Create a movieTitles RDD with (movie,title) tuple
moviesTitles = moviesRDD.map(lambda x: (int(x[0]),x[1]))
moviesTitles.take(3)
```




    [(1, 'Toy Story (1995)'),
     (2, 'Jumanji (1995)'),
     (3, 'Grumpier Old Men (1995)')]




```python
# Get just movie IDs for new user rated movies
newUserRatingsIds = map(lambda x: x[1], newUserRating) 

# Filter out complete movies RDD 
newUserUnratedRDD = (moviesRDD.filter(lambda x: x[0] not in newUserRatingsIds).map(lambda x: (newUserID, x[0])))

# Use the input RDD, new_user_unrated_movies_RDD, with new_ratings_model.predictAll() to predict new ratings for the movies
newRecRDD = newRatingsModel.predictAll(newUserUnratedRDD)

newRecRDD.take(3)
```




    [Rating(user=0, product=37739, rating=4.409561327443813),
     Rating(user=0, product=142192, rating=3.348429419520933),
     Rating(user=0, product=69069, rating=4.74908518246428)]



This new recommendation RDD `newRecRDD` now carries the predicted recommendations for new user. Now we can now look at top x number of movies with the highest predicted ratings and join these with the movies RDD to get the titles, and ratings count to make the results more meaningful. 

For this you need to perform following tasks:

* Map `newRecRDD` to build a (movie, ratings) tuple for each entry as `newRecRatingRDD`
* Use `.join()` transformation sequentially to to join `newRecRatingRDD` to `moviesTitles` and to `movieRatingsCountsRDD` to create 

A good resource on PySpark `.join()` is available at [THIS](http://www.learnbymarketing.com/1100/pyspark-joins-by-example/) resource.


```python
# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
newRecRatingRDD = newRecRDD.map(lambda x: (x.product, x.rating))
newRecRatingTitleCountRDD = newRecRatingRDD.join(moviesTitles)\
                                           .join(movieRatingsCountsRDD)
```

We can now simplify the the above to only include **(title, ratings, count)** and transform as a new RDD containing new ratings for unrated movies.


```python
newRecRatingTitleCountFlatRDD = newRecRatingTitleCountRDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
newRecRatingTitleCountFlatRDD.take(3)
```




    [('Rubber (2010)', 4.677853632654724, 1),
     ('Kate & Leopold (2001)', 4.58198704876763, 10),
     ('"Siege', 4.27845599256427, 16)]



FINALLY, we can get highest rated recommended movies for the new user, filtering out movies with less than 50 ratings (try changing this value).
For this we need to do following: 

* Use `.filter()` to include only the movies with more than 50 ratings.
* Use `.takeordered()` to get top 10 recommendations


```python
recommendations = newRecRatingTitleCountFlatRDD.filter(lambda x: x[2]>=50)
top10MoviesOrdered = recommendations.takeOrdered(10, key=lambda y: -y[1])

print ('TOP recommended movies (with more than 50 reviews):\n%s' %
        '\n'.join(map(str, top10MoviesOrdered)))


# TOP recommended movies (with more than 50 reviews):
# ('"Shawshank Redemption', 5.447804190989062, 311)
# ('V for Vendetta (2006)', 5.432833918216835, 73)
# ('Harry Potter and the Goblet of Fire (2005)', 5.424466636277112, 59)
# ('Life Is Beautiful (La Vita è bella) (1997)', 5.384201632659801, 99)
# ('"Lock', 5.380165378272083, 74)
# ('Forrest Gump (1994)', 5.337304995573618, 341)
# ('"Princess Bride', 5.328423741235671, 163)
# ('Good Will Hunting (1997)', 5.301483354034365, 157)
# ('"Breakfast Club', 5.234274895183267, 117)
# ('Slumdog Millionaire (2008)', 5.227081955573315, 52)
```

    TOP recommended movies (with more than 50 reviews):
    ('"Lord of the Rings: The Return of the King', 5.43193489557752, 176)
    ('"Shawshank Redemption', 5.421302815561621, 311)
    ('Life Is Beautiful (La Vita è bella) (1997)', 5.32921373025175, 99)
    ('To Kill a Mockingbird (1962)', 5.271840206257398, 80)
    ('"Lord of the Rings: The Two Towers', 5.270964795616836, 188)
    ('Harry Potter and the Goblet of Fire (2005)', 5.234257620475886, 59)
    ('Gladiator (2000)', 5.216978605187376, 161)
    ('"Dark Knight', 5.215975989082537, 121)
    ('"Lord of the Rings: The Fellowship of the Ring', 5.181898240172382, 200)
    ('"Lock', 5.181815189163409, 74)
    

Similarly, we can also check bottom 10 movies with lowest ratings with `.takeOrdered()`


```python
bottom10MoviesOrdered = recommendations.takeOrdered(10, key=lambda y: y[1])
print ('Lowest recommended movies (with more than 50 reviews):\n%s' %
        '\n'.join(map(str, bottom10MoviesOrdered)))

# Lowest recommended movies (with more than 50 reviews):
# ('Beverly Hills Cop III (1994)', 2.423247696283056, 57)
# ('"Blair Witch Project', 2.456475591917372, 86)
# ('Bowfinger (1999)', 2.495144318199298, 51)
# ('"Cable Guy', 2.633730093117032, 59)
# ('Congo (1995)', 2.784807232020519, 63)
# ('Species (1995)', 2.831861058132409, 55)
# ('Judge Dredd (1995)', 2.8391230652113846, 70)
# ('Mighty Aphrodite (1995)', 2.845570668091761, 51)
# ('Casper (1995)', 2.855333652701143, 58)
# ('Executive Decision (1996)', 3.0047635050446324, 61)
```

    Lowest recommended movies (with more than 50 reviews):
    ('"Cable Guy', 2.818825824040246, 59)
    ('"Blair Witch Project', 2.927431548859694, 86)
    ('Judge Dredd (1995)', 3.007265324895222, 70)
    ('Coneheads (1993)', 3.0383864093063018, 55)
    ('Bowfinger (1999)', 3.072113585498421, 51)
    ('Congo (1995)', 3.08657401179308, 63)
    ('Annie Hall (1977)', 3.1184840932911086, 80)
    ("Muriel's Wedding (1994)", 3.125778599638167, 56)
    ('Space Jam (1996)', 3.157511975071408, 50)
    ('American Psycho (2000)', 3.217647614386408, 51)
    

So here we have it. Our recommendation system is generating quite meaningful results with top 10 movies. A qualitative and subjective assessment shows top 10 movies are generally with a comedy/family/romance themes while the bottom 10 movies include some sci-fi/horror/action movies. 

## Next Steps

* Remember these results are only based on a subset of data. Run the code with complete dataset in a similar fashion and discuss the improvement and predictive performance through RMSE, as well as subjective evaluations based on your personal preferences / taste in movies. 

* Use movie genres/IMDB ratings found in large dataset as extra features for the model and inspect the improvement in model behaviour. 

#### Take it to the next level
* Use IMDB links to scrap user reviews from IMDB and using basic NLP techniques, create extra embeddings for ALS model. 



---

## Summary

In this lab, we learnt how to build a model using Spark, how to perform some parameter selection using a reduced dataset, and how to update the model every time that new user preferences come in. We looked at how Spark's ALS implementation can be be used to build a scalable and efficient reommendation system. We also saw that such systems can become computationaly expensive and using them with an online system could be a problem with traditional computational platforms. Spark's disctributed computing architecture provides a great solution to deploy such recommendation systems for real worls applications (think Amazon, Spotify).
