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
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

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
    
    kmModel.clusterCenters.map(vectorElement => new Pair(vectorElement(0), vectorElement(1)))
    
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
    
    ???
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    ???
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    ???
  }



}


