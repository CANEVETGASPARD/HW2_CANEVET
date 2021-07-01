from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, monotonically_increasing_id, col,collect_list

spark = SparkSession.builder.master("local[1]").appName('recommendation').getOrCreate()

movies_df = spark.read.csv("C/recommendation-datasets/movies.csv",inferSchema=True,header=True)
ratings_df = spark.read.csv("C/recommendation-datasets/ratings.csv",inferSchema=True,header=True)

# (i)
print("part i")
full_join_df = ratings_df.join(movies_df, ratings_df.movieId == movies_df.movieId,"inner")
full_join_df\
    .select(ratings_df.movieId,ratings_df.rating,movies_df.title)\
    .groupBy("title").agg(count("rating").alias("count_ratings"),\
                            avg("rating").alias("average_rating"))\
    .orderBy(["count_ratings","average_rating"],ascending=[0,0]).show(10)

# (ii)
#first looking for all genres and put them in a list
print("part ii")
GENRES = []
movies_genres_rdd = movies_df.rdd.flatMap(lambda row: row.genres.split("|")).map(lambda genre: (genre,1)).reduceByKey(lambda a,b: a+b).collect()
for row in movies_genres_rdd:
    GENRES.append(row[0])
print(GENRES)

#then display the top ten for each genre

#if we want to display only the better rated movies
for genre in GENRES:
    print(genre)
    full_join_df\
        .select(ratings_df.movieId,ratings_df.rating,movies_df.title,movies_df.genres)\
        .filter(movies_df.genres.like(f"%{genre}%"))\
        .groupBy("title").agg(count("rating").alias("count_ratings"),\
                                avg("rating").alias("average_rating"))\
        .orderBy(["average_rating"],ascending=[0]).show(10)

#if we want first to take in account the number of rating before ranking movies
for genre in GENRES:
    print(genre)
    full_join_df\
        .select(ratings_df.movieId,ratings_df.rating,movies_df.title,movies_df.genres)\
        .filter(movies_df.genres.like(f"%{genre}%"))\
        .groupBy("title").agg(count("rating").alias("count_ratings"),\
                                avg("rating").alias("average_rating"))\
        .orderBy(["count_ratings","average_rating"],ascending=[0,0]).show(10)

#(iii)
print("part iii")
movies_first100_df = movies_df.withColumn("row",monotonically_increasing_id()).filter(col("row") < 100).drop("row")
right_join_df = ratings_df.join(movies_first100_df, ratings_df.movieId == movies_df.movieId,"right")\
    .select(movies_first100_df.movieId,ratings_df.userId)
grouped_right_join_df = right_join_df.groupBy("userId").agg(collect_list("movieId").alias("moviesId")).show()
spark.stop()