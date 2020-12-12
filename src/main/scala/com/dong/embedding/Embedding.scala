package com.dong.embedding

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object Embedding {

    /**
     * 处理用户评分电影序列
     *
     * @param ratingSamples
     * @return
     */
    def processItemSequence(ratingSamples: DataFrame): RDD[Seq[String]] = {
        val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
            rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
                .sortBy { case (_, timestamp) => timestamp }
                .map { case (movieId, _) => movieId }
        })

        val userSeq = ratingSamples
            .where(col("rating") >= 3.5)
            .groupBy("userId")
            .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
            .withColumn("movieIdStr", array_join(col("movieIds"), " "))

        val movieSeq = userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
        movieSeq
    }


    /**
     * 训练 item2vec 模型
     *
     * @param samples
     * @param embLength
     * @return
     */
    def trainItem2vec(samples: RDD[Seq[String]], embLength: Int): Word2VecModel = {
        val word2vec = new Word2Vec()
            .setVectorSize(embLength)
            .setWindowSize(5)
            .setNumIterations(10)

        val model = word2vec.fit(samples)
        model
    }


    /**
     * 生成转移概率矩阵，以及各节点的分布
     *
     * @param samples
     * @return
     */
    def generateTransitionMatrix(samples: RDD[Seq[String]]): (mutable.Map[String, mutable.Map[String, Double]], mutable.Map[String, Double]) = {
        val pairSamples = samples.flatMap[(String, String)](sample => {
            var pairSeq = Seq[(String, String)]()
            var previousItem: String = null
            sample.foreach((element: String) => {
                if (previousItem != null) {
                    pairSeq = pairSeq :+ (previousItem, element)
                }
                previousItem = element
            })
            pairSeq
        })

        val pairCountMap = pairSamples.countByValue()
        var pairTotalCount = 0L
        val transitionCountMatrix = mutable.Map[String, mutable.Map[String, Long]]()
        val itemCountMap = mutable.Map[String, Long]()

        pairCountMap.foreach(pair => {
            val pairItems = pair._1
            val count = pair._2

            if (!transitionCountMatrix.contains(pairItems._1)) {
                transitionCountMatrix(pairItems._1) = mutable.Map[String, Long]()
            }

            transitionCountMatrix(pairItems._1)(pairItems._2) = count
            itemCountMap(pairItems._1) = itemCountMap.getOrElse[Long](pairItems._1, 0) + count
            pairTotalCount = pairTotalCount + count
        })

        val transitionMatrix = mutable.Map[String, mutable.Map[String, Double]]()
        val itemDistribution = mutable.Map[String, Double]()

        transitionCountMatrix foreach {
            case (itemAId, transitionMap) =>
                transitionMatrix(itemAId) = mutable.Map[String, Double]()
                transitionMap foreach { case (itemBId, transitionCount) => transitionMatrix(itemAId)(itemBId) = transitionCount.toDouble / itemCountMap(itemAId) }
        }

        itemCountMap foreach { case (itemId, itemCount) => itemDistribution(itemId) = itemCount.toDouble / pairTotalCount }
        (transitionMatrix, itemDistribution)
    }


    /**
     * 单次随机游走的过程
     *
     * @param transitionMatrix
     * @param itemDistribution
     * @param sampleLength
     * @return
     */
    def oneRandomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleLength: Int): Seq[String] = {
        val sample = mutable.ListBuffer[String]()

        //pick the first element
        val randomDouble = Random.nextDouble()
        var firstItem = ""
        var accumulateProb: Double = 0D
        breakable {
            for ((item, prob) <- itemDistribution) {
                accumulateProb += prob
                if (accumulateProb >= randomDouble) {
                    firstItem = item
                    break
                }
            }
        }

        sample.append(firstItem)
        var curElement = firstItem

        breakable {
            for (_ <- 1 until sampleLength) {
                if (!itemDistribution.contains(curElement) || !transitionMatrix.contains(curElement)) {
                    break
                }

                val probDistribution = transitionMatrix(curElement)
                val randomDouble = Random.nextDouble()
                breakable {
                    for ((item, prob) <- probDistribution) {
                        if (randomDouble >= prob) {
                            curElement = item
                            break
                        }
                    }
                }
                sample.append(curElement)
            }
        }
        Seq(sample.toList: _*)
    }


    /**
     * 随机游走，生成序列样本
     *
     * @param transitionMatrix
     * @param itemDistribution
     * @param sampleCount
     * @param sampleLength
     * @return
     */
    def randomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleCount: Int, sampleLength: Int): Seq[Seq[String]] = {
        val samples = mutable.ListBuffer[Seq[String]]()
        for (_ <- 1 to sampleCount) {
            samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
        }
        Seq(samples.toList: _*)
    }


    /**
     * 训练 Deep-Walk 模型生成图 embedding
     *
     * @param samples
     * @param sparkSession
     * @param embLength
     * @return
     */
    def graphEmb(samples: RDD[Seq[String]], sparkSession: SparkSession, embLength: Int): Word2VecModel = {
        val transitionMatrixAndItemDis = generateTransitionMatrix(samples)

        println(transitionMatrixAndItemDis._1.size)
        println(transitionMatrixAndItemDis._2.size)

        val sampleCount = 20000
        val sampleLength = 10
        val newSamples = randomWalk(transitionMatrixAndItemDis._1, transitionMatrixAndItemDis._2, sampleCount, sampleLength)

        val rddSamples = sparkSession.sparkContext.parallelize(newSamples)
        trainItem2vec(rddSamples, embLength)
    }


    /**
     * 生成用户的 embedding 向量
     *
     * @param ratingSamples
     * @param word2VecModel
     * @param embLength
     * @return
     */
    def generateUserEmb(ratingSamples: DataFrame, word2VecModel: Word2VecModel, embLength: Int): ArrayBuffer[(String, Array[Float])] = {
        val userEmbeddings = new ArrayBuffer[(String, Array[Float])]()

        // 将用户评价过的电影的 embedding 向量相加得到用户的 embedding 向量
        ratingSamples.collect().groupBy(_.getAs[String]("userId"))
            .foreach(user => {
                val userId = user._1
                var userEmb = new Array[Float](embLength)

                userEmb = user._2.foldRight[Array[Float]](userEmb)((row, newEmb) => {
                    val movieId = row.getAs[String]("movieId")
                    val movieEmb = word2VecModel.getVectors.get(movieId)
                    if (movieEmb.isDefined) {
                        newEmb.zip(movieEmb.get).map { case (x, y) => x + y }
                    } else {
                        newEmb
                    }
                })
                userEmbeddings.append((userId, userEmb))
            })

        userEmbeddings
    }


    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)

        val conf = new SparkConf()
            .setMaster("local")
            .setAppName("embedding")
            .set("spark.submit.deployMode", "client")
        val spark = SparkSession.builder.config(conf).getOrCreate()

        val ratingsResourcesPath = this.getClass.getResource("/webroot/ml-latest-small/ratings.csv")
        val ratingSamples = spark.read.format(source = "csv").option("header", "true").load(ratingsResourcesPath.getPath)
        println("原始的 rating 样本：")
        ratingSamples.printSchema()
        ratingSamples.show(10)

        val movieSeq = processItemSequence(ratingSamples)
        println("用户评分电影的序列：")
        println(movieSeq.take(10).foreach(println))

        // 用 item2vec 方法提取电影 embedding 向量
        val embLength = 10
        val item2VecModel = trainItem2vec(movieSeq, embLength)
        // 示例：id 为 158 的电影的 embedding 向量
        println(item2VecModel.getVectors("158").mkString(" "))
        // 示例：找出与 id 为 158 的电影相似的 20 部电影
        val synonyms = item2VecModel.findSynonyms("158", 20)
        for ((synonym, cosineSimilarity) <- synonyms) {
            println(s"$synonym $cosineSimilarity")
        }

        // 用 deep walk 方法提取电影的图 embedding 向量
        val deepWalkModel = graphEmb(movieSeq, spark, embLength)
        // 示例：id 为 158 的电影的 embedding 向量
        println(deepWalkModel.getVectors("158").mkString(" "))
        // 示例：找出与 id 为 158 的电影相似的 20 部电影
        val synonyms2 = deepWalkModel.findSynonyms("158", 20)
        for ((synonym, cosineSimilarity) <- synonyms2) {
            println(s"$synonym $cosineSimilarity")
        }

        // 通过电影 embedding 获取用户 embedding
        val userEmbeddings = generateUserEmb(ratingSamples, item2VecModel, embLength)
    }
}
