package com.madhukaraphatak.spark.ml.titanic
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.IntegerType

/**
 Random Forest for predicting survival in the titanic ship

 **/
object RandomForest {

  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    val df = sparkSession.read.option("header","true").option("inferSchema","true").csv("src/main/resources/titanic/train.csv")
    
    val genderStages = handleCategorical("Sex")
    val embarkedStages = handleCategorical("Embarked")
    val pClassStages = handleCategorical("Pclass")

    val cols = Array("Sex_onehot","Embarked_onehot","Pclass_onehot","SibSp","Parch","Age")
    val vectorAssembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")

    val randomForestClassifier = new RandomForestClassifier().setLabelCol("Survived")

    val preProcessStages = genderStages ++ embarkedStages ++ pClassStages ++ Array(vectorAssembler) 
    val pipeline = new Pipeline().setStages(preProcessStages ++ Array(randomForestClassifier))

    val meanValue = df.agg(mean(df("Age"))).first.getDouble(0)
    val fixedDf = df.na.fill(meanValue,Array("Age"))
    val dfs = fixedDf.randomSplit(Array(0.7,0.3))
    val trainDf = dfs(0)
    val crossDf = dfs(1)

    val model = pipeline.fit(trainDf)

    val testDf = sparkSession.read.option("header","true").option("inferSchema","true").csv("src/main/resources/titanic/test.csv")
    val fixedOutputDf = testDf.na.fill(meanValue,Array("age"))
    val scoredDf = model.transform(fixedOutputDf)

    scoredDf.printSchema()
    val outputDf = scoredDf.select("PassengerId","prediction")
    val castedDf = outputDf.select(outputDf("PassengerId"), outputDf("prediction").cast(IntegerType).as("Survived"))
    castedDf.write.format("csv").option("header","true").mode(SaveMode.Overwrite).save("src/main/resources/output/")
    //println(accuracyScore(predictDF,"Survived","prediction"))

  }
  def handleCategorical(column:String):Array[PipelineStage] = {
    val stringIndexer = new StringIndexer().setInputCol(column)
	    .setOutputCol(s"${column}_index")
	    .setHandleInvalid("skip")
    val oneHot = new OneHotEncoder().setInputCol(s"${column}_index").setOutputCol(s"${column}_onehot")
    Array(stringIndexer,oneHot)
  }
  
  def accuracyScore(df:DataFrame, label:String, predictCol:String) = {
    val totalValues = df.count()
    val matchingValues = df.select(label,predictCol).filter(row => {row.getInt(0).toDouble == row.getDouble(1)}).count
    matchingValues.toDouble / totalValues.toDouble
  }
}
