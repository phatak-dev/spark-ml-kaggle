package com.madhukaraphatak.spark.ml

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import MLUtils._

/**
 * Undersampling for Credict Card Fraud
 *
 */
object UnderSampling {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local[4]")
      .appName("example")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")
    //load train df
    val df = sparkSession.read.option("header", "true").option("inferSchema", "true").csv("src/main/resources/creditcard.csv")
    df.printSchema()

    val amountVectorAssembler = new VectorAssembler().setInputCols(Array("Amount")).setOutputCol("Amount_vector")
    val standarScaler = new StandardScaler().setInputCol("Amount_vector").setOutputCol("Amount_scaled")
    val dropColumns = Array("Time","Amount","Class")
    
    val cols = df.columns.filter( column => !dropColumns.contains(column)) ++ Array("Amount_scaled")
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")

    // pipeline 
    val logisticRegression = new LogisticRegression()
    val trainPipeline = new Pipeline().setStages(Array(amountVectorAssembler,standarScaler,vectorAssembler,logisticRegression))

    val dfs = df.randomSplit(Array(0.7, 0.3))
    val trainDf = dfs(0).withColumnRenamed("Class", "label")
    val crossDf = dfs(1)

    val imbalanceModel = trainPipeline.fit(trainDf)
    println("test accuracy with pipeline" + accuracyScore(imbalanceModel.transform(crossDf), "Class", "prediction"))
    println("test recall for 1.0 is " + recall(imbalanceModel.transform(crossDf), "Class", "prediction", 1.0))


  }
}
 
