package com.dong.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, sql}

object FeatureEngineering {

    /**
     * One-Hot 编码器
     * @param samples
     */
    def oneHotEncoder(samples:DataFrame): DataFrame = {
        val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))

        val oneHotEncoder = new OneHotEncoderEstimator()
            .setInputCols(Array("movieIdNumber"))
            .setOutputCols(Array("movieIdVector"))
            .setDropLast(false)

        val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
        oneHotEncoderSamples
    }


    val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }

    /**
     * Multi-Hot 编码器
     * @param samples
     * @return
     */
    def multiHotEncoder(samples:DataFrame): DataFrame = {
        val samplesWithGenres = samples.select(
            col("movieId"),
            col("title"),
            explode(split(col("genres"), "\\|").cast("array<string>")).as("genre")
        )
        val genreIndexer = new StringIndexer()
            .setInputCol("genre")
            .setOutputCol("genreIndex")
        val genreIndexSamples = genreIndexer
            .fit(samplesWithGenres)
            .transform(samplesWithGenres)
            .withColumn("genreIndexInt", col("genreindex").cast(sql.types.IntegerType))

        val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1
        val processedSample = genreIndexSamples
            .groupBy(col("movieId"))
            .agg(collect_list("genreIndexInt").as("genreIndexes"))
            .withColumn("indexSize", typedLit(indexSize))

        val finalSample = processedSample.withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))
        finalSample
    }


    val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

    /**
     * 处理 rating 样本中的数值型特征
     * @param samples
     * @return
     */
    def ratingFeatures(samples:DataFrame): DataFrame = {
        val movieFeatures = samples.groupBy(col("movieId"))
            .agg(
                count(lit(1)).as("ratingCount"),
                avg(col("rating")).as("avgRating"),
                variance(col("rating")).as("ratingVar")
            )
            .withColumn("avgRatingVec", double2vec(col("avgRating")))

        // 对电影的评论数进行分桶
        val ratingCountDiscretizer = new QuantileDiscretizer()
            .setInputCol("ratingCount")
            .setOutputCol("ratingCountBucket")
            .setNumBuckets(100)

        // 对电影评分的方差进行 MinMax 归一化
        val ratingScaler = new MinMaxScaler()
            .setInputCol("avgRatingVec")
            .setOutputCol("scaleAvgRating")

        val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
        val featurePipeline = new Pipeline().setStages(pipelineStage)

        val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
        movieProcessedFeatures
    }


    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)

        val conf = new SparkConf()
            .setMaster("local")
            .setAppName("featureEngineeringOneHot")
            .set("spark.submit.deployMode", "client")
        val spark = SparkSession.builder.config(conf).getOrCreate()

        val movieSamples = spark.read.format(source = "csv").option("header", "true").load("src/main/resources/ml-latest-small/movies.csv")
        val ratingSamples = spark.read.format(source = "csv").option("header", "true").load("src/main/resources/ml-latest-small/ratings.csv")

        println("原始的 movie 样本：")
        movieSamples.printSchema()
        movieSamples.show(10)

        val oneHotEncoderSamples = oneHotEncoder(movieSamples)
        println("One-Hot 编码后的 movie 样本：")
        oneHotEncoderSamples.printSchema()
        oneHotEncoderSamples.show(10)

        val multiHotEncoderSamples = multiHotEncoder(movieSamples)
        println("Multi-Hot 编码后的 movie 样本：")
        multiHotEncoderSamples.printSchema()
        multiHotEncoderSamples.show(10)

        val numericalProcessedSamples = ratingFeatures(ratingSamples)
        println("对评分进行数值型特征处理后的 movie 样本：")
        numericalProcessedSamples.printSchema()
        numericalProcessedSamples.show(10)
    }
}
