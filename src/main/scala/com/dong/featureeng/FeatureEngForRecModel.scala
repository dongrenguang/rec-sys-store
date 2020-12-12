package com.dong.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.net.URL
import scala.collection.immutable.ListMap
import scala.collection.mutable

object FeatureEngForRecModel {

    val NUMBER_PRECISION = 2

    /**
     * 添加电影特征
     * @param movieSamples
     * @param ratingSamplesWithLabel
     * @return
     */
    def addMovieFeatures(movieSamples: DataFrame, ratingSamples: DataFrame): DataFrame = {
        // 添加电影基本信息
        val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")

        // 添加电影发布年份
        val extractReleaseYearUdf = udf({(title: String) => {
           if (null == title || title.trim.length < 6
               || title.trim.substring(title.length - 6, title.length - 5) != "("
               || title.trim.substring(title.length - 1, title.length) != ")"
           ) {
               1990
           }
           else {
               title.trim.substring(title.length - 5, title.length - 1).toInt
           }
        }})
        val samplesWithMovies2 = samplesWithMovies1
            .withColumn("releaseYear", extractReleaseYearUdf(col("title")))
            .drop("title")

        // 添加电影风格
        val samplesWithMovies3 = samplesWithMovies2
            .withColumn("movieGenre1", split(col("genres"), "\\|").getItem(0))
            .withColumn("movieGenre2", split(col("genres"), "\\|").getItem(1))
            .withColumn("movieGenre3", split(col("genres"), "\\|").getItem(2))

        // 添加电影评分特征
        val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
            .agg(
                count(lit(1)).as("movieRatingCount"),
                format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
                stddev(col("rating")).as("movieRatingStddev")
            )
            .na.fill(0)
            .withColumn("movieRatingStddev", format_number(col("movieRatingStddev"), NUMBER_PRECISION))
        val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieId"), "left")
        samplesWithMovies4.show(10, truncate = false)

        samplesWithMovies4
    }


    // 电影风格提取
    val extractGenres: UserDefinedFunction = udf { (genreArray: Seq[String]) => {
        val genreMap = mutable.Map[String, Int]()
        genreArray.foreach((element:String) => {
            val genres = element.split("\\|")
            genres.foreach((oneGenre: String) => {
                genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0) + 1
            })
        })
        val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2): _*)
        sortedGenres.keys.toSeq
    }}


    /**
     * 添加用户特征
     * @param ratingSamples
     * @return
     */
    def addUserFeatures(ratingSamples: DataFrame): DataFrame = {
        val samplesWithUserFeatures = ratingSamples
            .withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieId")).otherwise(lit(null)))
                .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1))
            )
            .withColumn("userPositiveHistory", reverse(col("userPositiveHistory")))
            .withColumn("userRatedMovie1", col("userPositiveHistory").getItem(0))
            .withColumn("userRatedMovie2", col("userPositiveHistory").getItem(1))
            .withColumn("userRatedMovie3", col("userPositiveHistory").getItem(2))
            .withColumn("userRatedMovie4", col("userPositiveHistory").getItem(3))
            .withColumn("userRatedMovie5", col("userPositiveHistory").getItem(4))
            .withColumn("userRatingCount", count(lit(1))
                .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1))
            )
            .withColumn("userAvgReleaseYear", avg(col("releaseYear"))
                .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1)).cast(IntegerType)
            )
            .withColumn("userReleaseYearStddev", stddev(col("releaseYear"))
                .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1))
            )
            .withColumn("userAvgRating", format_number(avg(col("rating"))
                .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1)), NUMBER_PRECISION)
            )
            .withColumn("userRatingStddev", stddev(col("rating"))
                .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1))
            )
            .withColumn("userGenres", extractGenres(
                collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
                    .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1))
            ))
            .na.fill(0)
            .withColumn("userReleaseYearStddev", format_number(col("userReleaseYearStddev"), NUMBER_PRECISION))
            .withColumn("userRatingStddev", format_number(col("userRatingStddev"), NUMBER_PRECISION))
            .withColumn("userGenre1", col("userGenres").getItem(0))
            .withColumn("userGenre2", col("userGenres").getItem(1))
            .withColumn("userGenre3", col("userGenres").getItem(2))
            .withColumn("userGenre4", col("userGenres").getItem(3))
            .withColumn("userGenre5", col("userGenres").getItem(4))
            .drop("genres", "userGenres", "userPositiveHistory")
            .filter(col("userRatingCount") > 1)

        samplesWithUserFeatures.printSchema()
        samplesWithUserFeatures.show(10, truncate = false)

        samplesWithUserFeatures
    }


    /**
     * 提取电影特征
     * @param sample
     * @return
     */
    def extractMovieFeatures(samples: DataFrame): DataFrame = {
        val movieLatestSamples = samples
            .withColumn("movieRowNum", row_number().over(Window.partitionBy("movieId").orderBy(col("timestamp").desc)))
            .filter(col("movieRowNum") === 1)
            .select("movieId","releaseYear", "movieGenre1","movieGenre2","movieGenre3","movieRatingCount", "movieAvgRating", "movieRatingStddev")
            .na.fill("")

        movieLatestSamples.printSchema()
        movieLatestSamples.show(10, truncate = false)

        movieLatestSamples
    }


    /**
     * 提取用户特征
     * @param sample
     * @return
     */
    def extractUserFeatures(samples: DataFrame): DataFrame = {
        val userLatestSamples = samples
            .withColumn("userRowNum", row_number().over(Window.partitionBy("userId").orderBy(col("timestamp").desc)))
            .filter(col("userRowNum") === 1)
            .select("userId","userRatedMovie1", "userRatedMovie2","userRatedMovie3","userRatedMovie4","userRatedMovie5",
                "userRatingCount", "userAvgReleaseYear", "userReleaseYearStddev", "userAvgRating", "userRatingStddev",
                "userGenre1", "userGenre2","userGenre3","userGenre4","userGenre5")
            .na.fill("")

        userLatestSamples.printSchema()
        userLatestSamples.show(10, truncate = false)

        userLatestSamples
    }


    /**
     * 切分训练集和测试机
     * @param samples
     * @param sampleResourcesPath
     */
    def splitAndSaveTrainingTestSamples(samples: DataFrame, sampleResourcesPath: URL): Unit = {
        val Array(training, test) = samples.randomSplit(Array(0.8, 0.2))
        training.coalesce(1).write.option("header", "true").mode("overwrite")
            .csv(sampleResourcesPath + "/trainingSamples")
        test.coalesce(1).write.option("header", "true").mode("overwrite")
            .csv(sampleResourcesPath + "/testSamples")
    }


    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)

        val conf = new SparkConf()
            .setMaster("local")
            .setAppName("featureEngineering")
            .set("spark.submit.deployMode", "client")
        val spark = SparkSession.builder.config(conf).getOrCreate()

        val movieResourcesPath = this.getClass.getResource("/webroot/ml-latest-small/movies.csv")
        val movieSamples = spark.read.format(source = "csv").option("header", "true").load(movieResourcesPath.getPath)
        println("原始的 movie 样本：")
        movieSamples.printSchema()
        movieSamples.show(10)

        val ratingsResourcesPath = this.getClass.getResource("/webroot/ml-latest-small/ratings.csv")
        val ratingSamples = spark.read.format(source = "csv").option("header", "true").load(ratingsResourcesPath.getPath)
        println("原始的 rating 样本：")
        ratingSamples.printSchema()
        ratingSamples.show(10)

        val ratingSamplesWithLabel = ratingSamples.withColumn("label", when(col("rating") >= 3.5, 1).otherwise(0))
        ratingSamplesWithLabel.show(10)

        val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
        val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)

        val sampleResourcesPath = this.getClass.getResource("/webroot/ml-latest-small")
        samplesWithUserFeatures.coalesce(1).write.option("header", "true").mode("overwrite")
            .csv(sampleResourcesPath + "/modelSamples")
        splitAndSaveTrainingTestSamples(samplesWithUserFeatures, sampleResourcesPath)

        val movieFeatures = extractMovieFeatures(samplesWithUserFeatures)
        val userFeatures = extractUserFeatures(samplesWithUserFeatures)
    }
}
