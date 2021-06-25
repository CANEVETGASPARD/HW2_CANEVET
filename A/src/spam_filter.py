from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, rand
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer,StopWordsRemover
from model.PunctuationRemover import PunctuationRemover
import nltk
import numpy as np
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[1]").appName('spam_filter').getOrCreate()

# Load training and testing dataset
spam_training = spark.read.text("A/spam-datasets/spam_training.txt")
nospam_training = spark.read.text("A/spam-datasets/nospam_training.txt")
spam_testing = spark.read.text("A/spam-datasets/spam_testing.txt")
nospam_testing = spark.read.text("A/spam-datasets/nospam_testing.txt")

#add category column and change value column name
spam_training_formated = spam_training.withColumnRenamed("value","sentence").withColumn("category",lit("spam"))
nospam_training_formated = nospam_training.withColumnRenamed("value","sentence").withColumn("category",lit("nospam"))
spam_testing_formated = spam_testing.withColumnRenamed("value","sentence").withColumn("category",lit("spam"))
nospam_testing_formated = nospam_testing.withColumnRenamed("value","sentence").withColumn("category",lit("nospam"))

#group spam and nospam dataset from training and testing dataset
full_train_dataset = spam_training_formated.union(nospam_training_formated).orderBy(rand())
test_dataset = spam_testing_formated.union(nospam_testing_formated).orderBy(rand())

# Configure an ML pipeline, which consists of 6 stages: remove punctuation, tokenizer, remove stop words, hashingTF, idf and String Indexer.
nltk.download('stopwords')
stopWords = list(nltk.corpus.stopwords.words('english'))

punctuationRemover = PunctuationRemover(inputCol="sentence",outputCol="filteredSentence")
tokenizer = Tokenizer(inputCol=punctuationRemover.getOutputCol(), outputCol="words")
stopWordRemover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),outputCol="filtered_words",stopWords=stopWords)
hashingTF = HashingTF(inputCol=stopWordRemover.getOutputCol(), outputCol="tf")
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features",minDocFreq=5)
labelStringIndexer = StringIndexer(inputCol = "category", outputCol = "label")
pipeline = Pipeline(stages=[punctuationRemover,tokenizer,stopWordRemover,hashingTF,idf,labelStringIndexer])

#fit the first pipeline and transform test and train dataset -> column (sentence,category,words,tf,features,label)
NUMBEROFFEATURES = np.arange(1000,20000,1000)
print(NUMBEROFFEATURES)
ACCURACY = []

for nbrFeatures in NUMBEROFFEATURES:
    pipeline.getStages()[3].setNumFeatures(nbrFeatures)
    pipelineFitted = pipeline.fit(full_train_dataset)
    train_df = pipelineFitted.transform(full_train_dataset)
    test_df = pipelineFitted.transform(test_dataset)

    # logistic regression model fit with the dataset made with second pipeline
    lr = LogisticRegression(maxIter=10,featuresCol="features",labelCol="label")
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_df.count())
    print(f'Number of feature: {nbrFeatures}, accuracy: {accuracy}')
    ACCURACY.append(accuracy)

spark.stop()