from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder.master("local[1]").appName('mnist clustering').getOrCreate()

mnist_df = spark.read.csv("B/mnist-datasets/mnist_test.csv",inferSchema=True)

#if we want to see if there are some missing values
#from pyspark.sql.functions import isnan, when, count, col
#mnist_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in mnist_df.columns]).show()

vecAssembler = VectorAssembler(inputCols=mnist_df.columns , outputCol="features")
stdScalar = StandardScaler(inputCol="features", outputCol="scaledFeatures",withStd=True, withMean=True)
bikmeans = BisectingKMeans(maxIter =30,seed=1) #not used this time
kmeans = KMeans(maxIter =30,seed=1)
pipeline = Pipeline(stages=[vecAssembler,stdScalar,kmeans])

paramGrid = ParamGridBuilder()\
    .addGrid(kmeans.k, [7,8,9,10,11,12]) \
    .build()

tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=ClusteringEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(mnist_df)
print(f"best number of cluster k: {model.bestModel.stages[2].getK()}")
centers = model.bestModel.stages[2].clusterCenters()

np.save("B/centroids",centers)

spark.stop()