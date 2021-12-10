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

import breeze.linalg._
import breeze.numerics._
import breeze.plot._

object assignment  {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)


  val spark = SparkSession.builder()
                          .appName("assignment")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()

  // Define schemas for the dataframes and read the data
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

  // Create dataframe with labels mapped to numerical values, Ok to 0 and Fatal to 1
  val dataK5D3WithLabels = dataK5D2.withColumn("num(LABEL)", when(dataK5D2("LABEL") === " Ok", 0)
                                    .otherwise(1))
                                    
  dataK5D3WithLabels.show


  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("unscaledFeatures")
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    // Make a transformation pipeline to first use vectorAssembler and then scale the data
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)
    
    val kmModel = kmeans.fit(transformedData)
    
    kmModel.summary.predictions.show
    
    // Map the vector elements to an array to create tuples
    kmModel.clusterCenters.map(vectorElement => (vectorElement(0), vectorElement(1)))
    
  }

  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("unscaledFeatures")
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    // Make a transformation pipeline to first use vectorAssembler and then scale the data
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)
    
    val kmModel = kmeans.fit(transformedData)
    
    kmModel.summary.predictions.show
    
    // Map the vector elements to an array to create triples
    kmModel.clusterCenters.map(vectorElement => (vectorElement(0), vectorElement(1), vectorElement(2)))
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "num(LABEL)"))
      .setOutputCol("unscaledFeatures")
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    // Make a transformation pipeline to first use vectorAssembler and then scale the data
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)
    
    val kmModel = kmeans.fit(transformedData)
    
    kmModel.summary.predictions.show
    
    // Sort the cluster centers to decreasing order by the third value, as higher value means more fatal results,
    // and then map the first two elements of the first two vectors to tuples to an array
    kmModel.clusterCenters
      .sortWith((vectorElement1, vectorElement2)=>vectorElement1(2) > vectorElement2(2))
      .take(2)
      .map(vectorElement => (vectorElement(0), vectorElement(1)))
    
  }

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("unscaledFeatures")
    
    val scaler = new MinMaxScaler()
      .setInputCol("unscaledFeatures")
      .setOutputCol("features")
    
    // Make a transformation pipeline to first use vectorAssembler and then scale the data
    val transformationPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, scaler))
    
    //Fit produces a transformer
    val pipeLine = transformationPipeline.fit(df)
    val transformedData = pipeLine.transform(df)
    transformedData.show
    
    // Compute the cost array recursively, print it out and return it
    val costsArray = recursiveCosts(transformedData, high, low)
    costsArray.foreach(println)
    
    // Make graph of the elbow
    val fig = Figure()
    val p = fig.subplot(0)
    val cost = costsArray.map(element => element._2)
    val clusterAmount = costsArray.map(element => element._1.toDouble)
    p += plot(clusterAmount, cost) 
    scala.io.StdIn.readLine()
    
    return costsArray
  }
  
  /* Parameter df is the transformed data frame, k is desired k value for KMeans and hlow is the lowest required k
   * The costs are added to the array from the back, hence giving the array as parameter to the recursion is not necessary
   */
  def recursiveCosts(df: DataFrame, k: Int, low: Int): Array[(Int, Double)] = {
    // Create the KMeans for the given seed
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)

    val kmModel = kmeans.fit(df)
    val cost = kmModel.computeCost(df)
    
    /* If the current k is the highest required, add the cost to an empty array and return it
    * Otherwise enter new recursion with a k one higher and add the cost of this recursion level to the returned value
    */
    if (k == low) {
      return Array() :+ (k, cost)
    }
    else {
      return recursiveCosts(df, k - 1, low) :+ (k, cost)
    }
  }

}


