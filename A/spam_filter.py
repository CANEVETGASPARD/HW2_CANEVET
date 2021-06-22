from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, rand
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer

spark = SparkSession.builder.master("local[1]").appName('spam_filter').getOrCreate()

# Load training and testing dataset
spam_training = spark.read.text("spam-datasets/spam_training.txt")
nospam_training = spark.read.text("spam-datasets/nospam_training.txt")
spam_testing = spark.read.text("spam-datasets/spam_testing.txt")
nospam_testing = spark.read.text("spam-datasets/nospam_testing.txt")

#add category column and change value column name
spam_training_formated = spam_training.withColumnRenamed("value","sentence").withColumn("category",lit("spam"))
nospam_training_formated = spam_training.withColumnRenamed("value","sentence").withColumn("category",lit("nospam"))
spam_testing_formated = spam_testing.withColumnRenamed("value","sentence").withColumn("category",lit("spam"))
nospam_testing_formated = spam_testing.withColumnRenamed("value","sentence").withColumn("category",lit("nospam"))

#group spam and nospam dataset from training and testing dataset
train_dataset = spam_training_formated.union(nospam_training_formated).orderBy(rand())
test_dataset = spam_testing_formated.union(nospam_testing_formated).orderBy(rand())

# Configure an ML pipeline, which consists of 4 stages: tokenizer, hashingTF, idf and String Indexer.
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="tf",numFeatures=2000)
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",minDocFreq=5)
labelStringIndexer = StringIndexer(inputCol = "category", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer,hashingTF,idf,labelStringIndexer])

# Configure an other ML pipeline, which consists of 4 stages: tokenizer, countvectorizer, idf and String Indexer.
cv = CountVectorizer(vocabSize=2000, inputCol=tokenizer.getOutputCol(), outputCol='cv')
idfCv = IDF(inputCol=cv.getOutputCol(),outputCol="features",minDocFreq=5)
pipelineCv = Pipeline(stages=[tokenizer, cv, idfCv, labelStringIndexer])

#fit the first pipeline and transform test and train dataset -> column (sentence,category,words,tf,features,label)
pipelineFitted = pipeline.fit(train_dataset)
train_df = pipelineFitted.transform(train_dataset)
test_df = pipelineFitted.transform(test_dataset)

#fit the second pipeline and transform test and train dataset -> column (sentence,category,words,tf,features,label)
pipelineCvFitted = pipelineCv.fit(train_dataset)
trainCv_df = pipelineCvFitted.transform(train_dataset)
testCv_df = pipelineCvFitted.transform(test_dataset)

#print schema of each train dataset
train_df.printSchema()
trainCv_df.printSchema()

#show some rows of trainCv dataset
trainCv_df.show()

# logistic regression model fit with the dataset made with second pipeline
lr = LogisticRegression(maxIter=10,featuresCol="features",labelCol="label")
model = lr.fit(trainCv_df)
predictions = model.transform(test_df)

#print coefficient -> now they are empty
print(model.coefficients)

#accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_df.count())
#print(accuracy)

spark.stop()