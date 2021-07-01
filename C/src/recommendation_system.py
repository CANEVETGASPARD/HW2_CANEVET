from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

spark = SparkSession.builder.master("local[1]").appName('recommendationTop10').getOrCreate()

#load data
ratings_df = spark.read.csv("C/recommendation-datasets/ratings.csv",inferSchema=True,header=True)
ratingsRDD = ratings_df.rdd.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=10, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

paramGrid = ParamGridBuilder()\
    .addGrid(als.regParam, [0.1,0.05,0.01]) \
    .build()


# Evaluate the model by using a trainValidationSplit method
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

tvs = TrainValidationSplit(estimator=als,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(training)
predictions = model.transform(test)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# Generate top 10 movie recommendations for each user
userRecs = model.bestModel.recommendForAllUsers(10)
userRecs.show(truncate=False)

spark.stop()