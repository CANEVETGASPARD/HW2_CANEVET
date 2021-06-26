from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
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
pipeline = Pipeline(stages=[vecAssembler,stdScalar])

pipelineFitted = pipeline.fit(mnist_df)
mnist_df_scaled = pipelineFitted.transform(mnist_df)

K = [4,5,6,7,8,9,10,11,12,13]


"""for k in K:
    kmeans = KMeans().setK(k).setMaxIter(10).setSeed(1).setFeaturesCol("scaledFeatures")
    model = kmeans.fit(mnist_df_scaled)
    # Make predictions
    predictions = model.transform(mnist_df_scaled)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print(f"For k = {k}, silhouette with squared euclidean distance = {silhouette}")"""

kmeans = KMeans().setK(10).setMaxIter(30).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(mnist_df_scaled)
centers = model.clusterCenters()

np.save("B/centroidsNonScaled",centers)

spark.stop()