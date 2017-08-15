// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.io.File

import com.microsoft.CNTK.{DataType => CNTKDataType, Function => CNTKFunction, _}
import com.microsoft.ml.spark.schema.DatasetExtensions
import org.apache.commons.io.FileUtils._
import org.apache.spark.broadcast._
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._

private object CNTKModelUtils extends java.io.Serializable {

  private def applyModel(inputColInds: Array[Int],
                         broadcastModelBytes: Broadcast[Array[Byte]],
                         minibatchSize: Int,
                         inputNodeInds: Array[Int],
                         outputNodes: Option[Array[String]])(inputRows: Iterator[Row]): Iterator[Row] = {
    val device = DeviceDescriptor.useDefaultDevice
    val m      = CNTKModel.loadModelFromBytes(broadcastModelBytes.value, device)
    val model = outputNodes
      .map { names =>
        if (names.length == 1)
          CNTKLib.AsComposite(
            Option(m.findByName(names.head))
              .getOrElse(throw new IllegalArgumentException(s"Node $names does not exist")))
        else m
      }
      .getOrElse(m)

    val inputVars = inputNodeInds.map(model.getArguments.get(_))
    require(inputVars.forall(_.getDataType() == CNTKDataType.Float),
            "all input variable types are not Float")
    val inputShapes = inputVars.map(_.getShape)

    // This defines and instantiates an iterator, hasNext and next are the abstract methods that
    // define the interface and inputBuffer and outputBuffer hold the input and output rows so that
    // they can be joined and returned.
    // The logic inside next checks to see if the buffer is empty, and if so sends a new batch off
    // to be evaluated
    new Iterator[Row] {
      val inputBuffer      = new ListBuffer[Row]()
      val outputBuffer     = new ListBuffer[Row]()
      val inputVectorSizes = inputShapes.map(_.getTotalSize().toInt)
      val fvs: Array[Array[FloatVector]] =
        inputVectorSizes.map(n => Array.fill(minibatchSize)(new FloatVector(n.toLong)))
      val inputFVVs: Array[FloatVectorVector] =
        Array.fill(inputVars.size)(new FloatVectorVector(minibatchSize.toLong))

      def hasNext: Boolean = inputRows.hasNext || outputBuffer.nonEmpty

      def next(): Row = {
        if (outputBuffer.isEmpty) {
          var paddedRows = 0

          for ((colInd, i) <- inputColInds.zipWithIndex) {
            for (j <- 0 until minibatchSize) {
              if (inputRows.hasNext) {
                val row = inputRows.next()
                inputBuffer += row
                for ((x, k) <- row.getSeq[Float](colInd).view.zipWithIndex) {
                  fvs(i)(j).set(k, x)
                }
              } else {
                // TODO remove padding after CNTK bug is fixed
                paddedRows += 1
                for (k <- 0 until inputVectorSizes(i)) {
                  fvs(i)(j).set(k, 0.0.toFloat)
                }
              }
              inputFVVs(i).set(j, fvs(i)(j))
            }
          }

          val inputVals = (inputShapes zip inputFVVs).map {
            case (shp, fvv) => Value.createDenseFloat(shp, fvv, device)
          }
          val inputDataMap = new UnorderedMapVariableValuePtr()
          (inputVars zip inputVals).foreach { case (vari, value) => inputDataMap.add(vari, value) }

          val outputDataMap = new UnorderedMapVariableValuePtr()
          val outputVars    = model.getOutputs
          outputVars.foreach(outputDataMap.add(_, null))

          model.evaluate(inputDataMap, outputDataMap, device)

          val outputFVVs = Array.fill(outputVars.size)(new FloatVectorVector())
          (outputVars zip outputFVVs).foreach {
            case (vari, vect) => outputDataMap.getitem(vari).copyVariableValueToFloat(vari, vect)
          }
          assume(outputBuffer.isEmpty,
                 "The output row buffer was not empty when new elements were being added.")
          val outputSeqVecs = outputFVVs.map(fvv => toSeqSeq(fvv).dropRight(paddedRows)
                                                                 .map(fv => Vectors.dense(fv.map(_.toDouble).toArray)))
          val actualBatchSize = minibatchSize - paddedRows
          val unzippedBatches = for (i <- 0 until actualBatchSize) yield outputSeqVecs.map(_.apply(i))
          outputBuffer ++= unzippedBatches.map(Row.fromSeq(_))
        }
        val ret = Row.merge(inputBuffer.head, outputBuffer.head)
        inputBuffer.remove(0)
        outputBuffer.remove(0)
        ret
      }
    }
  }

  // here just for serialization
  val applyModelFunc = (inputColInds: Array[Int],
                        broadcastModelBytes: Broadcast[Array[Byte]],
                        minibatchSize: Int,
                        inputNodeInds: Array[Int],
                        outputNodes: Option[Array[String]]) => { (inputRows: Iterator[Row]) =>
    applyModel(inputColInds, broadcastModelBytes, minibatchSize, inputNodeInds, outputNodes)(inputRows)
  }

  private def toSeqSeq(fvv: FloatVectorVector): Seq[Seq[Float]] = {
    (0 until fvv.size.toInt).map(i => (0 until fvv.get(i).size.toInt).map(j => fvv.get(i).get(j)))
  }
}

object CNTKModel extends ComplexParamsReadable[CNTKModel] {
  def loadModelFromBytes(bytes: Array[Byte],
                         device: DeviceDescriptor =
                           DeviceDescriptor.useDefaultDevice): CNTKFunction = {
    import java.util.UUID._
    val modelFile = new File(s"$getTempDirectoryPath/$randomUUID.model")
    writeByteArrayToFile(modelFile, bytes)
    val model = try {
      CNTKFunction.load(modelFile.getPath, device)
    } finally forceDelete(modelFile)
    model
  }

}

@InternalWrapper
class CNTKModel(override val uid: String) extends Model[CNTKModel]
    with HasInputCol  with HasInputCols
    with HasOutputCol with HasOutputCols
    with ComplexParamsWritable with Wrappable {

  def this() = this(Identifiable.randomUID("CNTKModel"))

  /** Array of bytes containing the serialized CNTK <code>Function</code>
    * @group param
    */
  val model: ByteArrayParam =
    new ByteArrayParam(this, "model", "Array of bytes containing the serialized CNTKModel")

  /** @group setParam */
  def setModel(bytes: Array[Byte]): this.type = set(model, bytes)

  /** @group getParam */
  def getModel: Array[Byte] = $(model)

  /** @group setParam */
  def setModelLocation(spark: SparkSession, path: String): this.type = {
    val modelBytes = spark.sparkContext.binaryFiles(path).first()._2.toArray
    setModel(modelBytes)
  }

  /** Index of the input node
    * @group param
    */
  val inputNodes: IntArrayParam = new IntArrayParam(this, "inputNode", "index of the input node")

  /** @group setParam */
  def setInputNode(value: Int): this.type = set(inputNodes, Array(value))

  /** @group setParam */
  def setInputNodes(value: Array[Int]): this.type = set(inputNodes, value)

  /** @group getParam */
  def getInputNode: Int = {
    if (getInputNodes.length == 1) getInputNodes.head
    else throw new Exception("Must have one and only one inputNode set in order to getInputNode")
  }

  /** @group getParam */
  def getInputNodes: Array[Int] = $(inputNodes)

  /** Index of the output node
    * @group param
    */
  val outputNodeIndices: IntArrayParam =
    new IntArrayParam(this, "outputNodeIndices", "index of the output node")

  /** @group setParam */
  def setOutputNodeIndices(value: Array[Int]): this.type = set(outputNodeIndices, value)

  /** @group getParam */
  def getOutputNodeIndices: Array[Int] = $(outputNodeIndices)

  /** Name of the output node
    * @group param
    */
  val outputNodeNames: StringArrayParam =
    new StringArrayParam(this, "outputNodeNames", "name of the output node")

  /** @group setParam */
  def setOutputNodeNames(value: Array[String]): this.type = set(outputNodeNames, value)

  /** @group getParam */
  def getOutputNodeNames: Array[String] = $(outputNodeNames)

  /** Size of minibatches. Must be greater than 0; default is 10
    * @group param
    */
  val miniBatchSize: IntParam =
    new IntParam(this, "miniBatchSize", "size of minibatches", ParamValidators.gt(0))

  /** @group setParam */
  def setMiniBatchSize(value: Int): this.type = set(miniBatchSize, value)

  /** @group getParam */
  def getMiniBatchSize: Int = $(miniBatchSize)
  setDefault(miniBatchSize -> 10)

  /** @group setParam */
  override def setOutputCol(value: String): this.type = super.setOutputCols(Array(value))

  /** @group getParam */
  override def getOutputCol: String = {
    if (getOutputCols.length == 1) getOutputCols.head
    else throw new Exception("Must have one and only one outputCol set in order to getOutputCol")
  }

  /** @group setParam */
  override def setInputCol(value: String): this.type = super.setInputCols(Array(value))

  /** @group getParam */
  override def getInputCol: String = {
    if (getInputCols.length == 1) getInputCols.head
    else throw new Exception("Must have one and only one inputCol set in order to getInputCol")
  }

  def transformSchema(schema: StructType): StructType =
    getOutputCols.foldLeft(schema)((sch, col) => sch.add(StructField(col, VectorType)))

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  /** Evaluate the model
    * @param dataset the dataset to featurize
    * @return featurized dataset
    */
  def transform(dataset: Dataset[_]): DataFrame = {
    val spark        = dataset.sparkSession
    val sc           = spark.sparkContext
    val inputIndices = getInputCols.map(dataset.columns.indexOf(_))
    val missingCols =
      inputIndices.zip(getInputCols).filter { case (ind, col) => ind == -1 }.map(_._2)
    val device = DeviceDescriptor.useDefaultDevice

    require(missingCols.isEmpty, s"Input columns ${missingCols.mkString(", ")} do not exist")

    val model = CNTKModel.loadModelFromBytes(getModel, device)

    val setByName  = get(outputNodeNames)
    val setByIndex = get(outputNodeIndices)

    val outputNodes: Option[Array[String]] =
      if (setByName.isDefined) setByName
      else                     setByIndex.map(_.map(model.getOutputs.get(_).getName))

    val coersionOptionUDFs = inputIndices.map {
      dataset.schema.fields(_).dataType match {
        case ArrayType(tp, _) =>
          tp match {
            case DoubleType => Some(udf((x: mutable.WrappedArray[Double]) => x.map(_.toFloat)))
            case FloatType  => None
            case _ =>
              throw new IllegalArgumentException(s"improper column type: $tp, need Array[Float]")
          }
        case VectorType => Some(udf((x: DenseVector) => x.toArray.map(_.toFloat)))
      }
    }

    val (df, selectedIndices, coercedCols) = (coersionOptionUDFs zip inputIndices).foldLeft(
      (dataset.toDF, Array[Int](), Array[String]()))(
      (workDFAndOutputIndsAndPreviouslyCoerced, optionUDFAndInputInd) => {
        val (workDF, outputInds, previouslyCoerced) = workDFAndOutputIndsAndPreviouslyCoerced
        val (optionUDF, inputInd) = optionUDFAndInputInd
        optionUDF match {
          case Some(coersionUDF) => {
            val coercedCol = DatasetExtensions.findUnusedColumnName("coerced")(workDF.columns.toSet)
            val coercedDF = workDF.withColumn(coercedCol, coersionUDF(col(workDF.columns(inputInd))))
            (coercedDF, outputInds :+ workDF.columns.size, previouslyCoerced :+ coercedCol)
          }
          case None => (workDF, outputInds :+ inputIndices(outputInds.size), previouslyCoerced)
        }
      })

    val inputType           = df.schema($(inputCols).head).dataType
    val broadcastModelBytes = sc.broadcast(getModel)
    setDefault(inputNodes -> Array.range(0, model.getArguments.size - 1))
    val rdd = df.rdd.mapPartitions(
      CNTKModelUtils.applyModelFunc(selectedIndices,
                                    broadcastModelBytes,
                                    getMiniBatchSize,
                                    getInputNodes,
                                    outputNodes))
    setDefault(outputCols -> model.getOutputs.map(_.getName).toArray) // defaults to all CNTK model outputs
    if (setByName.isDefined && setByIndex.isDefined)
      throw new Exception("Must specify neither or only one of outputNodeName or outputNodeIndices")
    val output = spark.createDataFrame(rdd, transformSchema(df.schema))

    coercedCols.foldLeft(output)((_df, col) => _df.drop(col))
  }

}
