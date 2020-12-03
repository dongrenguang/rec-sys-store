package com.dong.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object OneHot {
    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)

        val conf = new SparkConf()
            .setMaster("local")
            .setAppName("featureEngineeringOneHot")
            .set("spark.submit.deployMode", "client")

        val spark = SparkSession.builder.config(conf).getOrCreate()
        val movie = spark.read.format(source = "csv").option("header", "true").load("src/main/resources/ml-latest-small/movies.csv")
        println(movie.printSchema())
    }
}
