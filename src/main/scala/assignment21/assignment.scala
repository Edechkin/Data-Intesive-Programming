package assignment21

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg, when}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.{VectorAssembler, MinMaxScaler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range
import com.sun.org.apache.xalan.internal.xsltc.compiler.ForEach

import scala.util.Sorting


object assignment  {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)


  val spark = SparkSession.builder()
                          .appName("assignment")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()

  val schema1 = new StructType()
    .add(StructField("a", DoubleType, true))
    .add(StructField("b", DoubleType, true))
    .add(StructField("LABEL", StringType, true))

  val schema2 = new StructType()
    .add(StructField("a", DoubleType, true))
    .add(StructField("b", DoubleType, true))
    .add(StructField("c", DoubleType, true))
    .add(StructField("LABEL", StringType, true))

  val dataK5D2 =  spark.read
                       .option("header", "true")
                       .schema(schema1)
                       .csv("data/dataK5D2.csv")

  val dataK5D3 =  spark.read
                       .option("header", "true")
                       .schema(schema2)
                       .csv("data/dataK5D3.csv")
                       
  val dataK5D3WithLabels = dataK5D2.withColumn("num(LABEL)", when(dataK5D2("LABEL") === " Ok", 0)
                                    .otherwise(1))
                                    
  dataK5D3WithLabels.show


  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("unscaledFeatures")
    
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    val scalerModel = scaler.fit(transformedData)
    
    val scaledData = scalerModel.transform(transformedData)
    
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)
    
    val kmModel = kmeans.fit(scaledData)
    
    kmModel.summary.predictions.show
    
    kmModel.clusterCenters.map(vectorElement => (vectorElement(0), vectorElement(1)))
    
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("unscaledFeatures")
    
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    val scalerModel = scaler.fit(transformedData)
    
    val scaledData = scalerModel.transform(transformedData)
    
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)
    
    val kmModel = kmeans.fit(scaledData)
    
    kmModel.summary.predictions.show
    
    kmModel.clusterCenters.map(vectorElement => (vectorElement(0), vectorElement(1), vectorElement(2)))
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "num(LABEL)"))
      .setOutputCol("unscaledFeatures")
    
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    val scalerModel = scaler.fit(transformedData)
    
    val scaledData = scalerModel.transform(transformedData)
    
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)
    
    val kmModel = kmeans.fit(scaledData)
    
    kmModel.summary.predictions.show
    
    kmModel.clusterCenters
      .sortWith((vectorElement1, vectorElement2)=>vectorElement1(2) > vectorElement2(2))
      .map(vectorElement => (vectorElement(0), vectorElement(1)))
      .take(2)
    
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("unscaledFeatures")
    
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    val scalerModel = scaler.fit(transformedData)
    
    val scaledData = scalerModel.transform(transformedData)
    
    var arrayOfCosts: Array[(Int, Double)] = Array()
    
    for ( k <- low to high){
    
      val kmeans = new KMeans()
        .setK(k)
        .setSeed(1L)
    
      val kmModel = kmeans.fit(scaledData)
      val cost = kmModel.computeCost(scaledData)
      arrayOfCosts +:= (k, cost)
    }
    return arrayOfCosts
  }

}


