package org.apache.spark.ml.linregr

import breeze.linalg.DenseVector
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasMaxIter, HasStepSize, HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.types.{StructType}
import org.apache.spark.sql.functions.{col, udf}

trait LinearRegressionParams extends PredictorParams with HasMaxIter with HasStepSize
  with HasFeaturesCol with HasLabelCol with HasPredictionCol {
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  setDefault(
    maxIter -> 1000,
    stepSize -> 0.001
  )

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkNumericType(schema, getLabelCol)
    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
    schema
  }
}


class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val featureEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val labelsEncoder: Encoder[Double] = ExpressionEncoder()
    val data: Dataset[(Vector, Double)] = dataset.select(
      col($(featuresCol)).as[Vector],
      col($(labelCol)).as[Double])
    val numFeatures: Int = data.first()._1.size
    val numRows = data.rdd.count()
    var weights = DenseVector.fill(numFeatures, 0.0)
    var bias: Double = 0.0

    for (iteration <- 1 to getMaxIter){
      val (gradW, gradB) = data.rdd.mapPartitions { iterator: Iterator[(Vector, Double)] =>
        var WgradSum = Vectors.zeros(numFeatures).asBreeze
        var BgradSum = 0.0
        iterator.foreach { case (features, label) =>
          val X = features.asBreeze
          val predicted = X.dot(weights) + bias
          val error = predicted - label
          WgradSum += X * error
          BgradSum += error
        }
        Iterator((WgradSum, BgradSum))
      }.reduce{case ((gradW1, gradB1), (gradW2, gradB2)) =>
        (gradW1 + gradW2, gradB1 + gradB2)}
      weights -= gradW * (getStepSize / numRows)
      bias -= gradB * (getStepSize / numRows)
    }
    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights).toDense, bias)).setParent(this)
  }
  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}


class LinearRegressionModel (override val uid: String,
                              val weights: Vector,
                              val bias: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams {


  private[linregr] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf((features: Vector) => {
      Vectors.fromBreeze(
        breeze.linalg.DenseVector(weights.asBreeze.dot(features.asBreeze)) + bias
      )
    })
    dataset.withColumn("prediction", predictUDF(col($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}