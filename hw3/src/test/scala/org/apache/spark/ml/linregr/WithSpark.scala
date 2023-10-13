package org.apache.spark.ml.linregr


import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
  lazy val sparkSession: SparkSession = WithSpark.sparkSession
  lazy val sqlContext: SQLContext = WithSpark.sqlContext
}

object WithSpark {
  lazy val sparkSession: SparkSession = SparkSession.builder
    .appName("hw_3")
    .master("local[4]")
    .getOrCreate()

  sparkSession.sparkContext.setLogLevel("ERROR")

  lazy val sqlContext: SQLContext = sparkSession.sqlContext
}