package org.apache.spark.ml.linregr

import org.scalatest.flatspec._
import org.scalatest.matchers._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import breeze.linalg.{DenseMatrix, DenseVector, *}
import org.apache.spark.ml.linalg
import breeze.stats.mean

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val eps = 1e-7
  val totMSE = 1e-10

  lazy val dataset: DataFrame = LinearRegressionTest._dataset
  lazy val weights: linalg.DenseVector = LinearRegressionTest._weights
  lazy val bias: Double = 0.0
  lazy val trueLabels: DenseVector[Double] = LinearRegressionTest._y

  val stepSize = 1.0
  val maxIter = 400

  var fitWeights: Vector = _
  var fitBias: Double = _

  "Fit" should "find correct weights" in {
    val estimator: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMaxIter(maxIter)
      .setStepSize(stepSize)

    val model = estimator.fit(dataset)

    println(s"Found weights: ", model.weights(0), model.weights(1), model.weights(2), model.bias)
    model.weights(0) shouldEqual (weights(0) +- eps)
    model.weights(1) shouldEqual (weights(1) +- eps)
    model.weights(2) shouldEqual (weights(2) +- eps)
    model.bias shouldEqual (bias +- eps)

    fitWeights = model.weights
    fitBias = model.bias

  }

  "Transform" should "make prediction with small mse" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = fitWeights,
      bias = fitBias
    ).setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    val prediction = model.transform(dataset)
      .select("prediction")
      .collect()
      .map { row => row.getAs[Vector](0)(0) }
    val errors = DenseVector(prediction) - trueLabels
    val mse: Double = mean(errors.map(x => x * x))
    println(s"Total MSE: $mse")
    mse should be(0.0 +- totMSE)
  }

}


object LinearRegressionTest extends WithSpark {

  lazy val _weights: linalg.DenseVector = Vectors.dense(1.5, 0.3, -0.7).toDense
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
  lazy val _y: DenseVector[Double] = _X * _weights.asBreeze
  lazy val _data: DenseMatrix[Double] = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)

  lazy val _dataset: DataFrame = {
    import sqlContext.implicits._

    val df = _data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")
    val output = assembler.transform(df).select("features", "y")
    output
  }
}