// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import com.microsoft.CNTK.CNTKExtensions._
import com.microsoft.CNTK.{DataType => CNTKDataType, SerializableFunction => CNTKFunction, _}
import com.microsoft.ml.spark.schema.DatasetExtensions
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

  def applyCNTKFunction(model: CNTKFunction,
    inputFVVs: Array[FloatVectorVector],
    inputVars: Array[Variable],
    device: DeviceDescriptor): Array[FloatVectorVector] = {
      val inputShapes  = inputVars.map(_.getShape)
      val inputVals = (inputShapes zip inputFVVs).map {
        case (shp, fvv) => Value.createDenseFloat(shp, fvv, device)
      }
      val inputDataMap  = new UnorderedMapVariableValuePtr()
      (inputVars zip inputVals).foreach { case (vari, value) => inputDataMap.add(vari, value) }

      val outputDataMap = new UnorderedMapVariableValuePtr()
      val outputVars    = model.getOutputs
      outputVars.foreach(outputDataMap.add(_, null))

      model.evaluate(inputDataMap, outputDataMap, device)

      val outputFVVs = Array.fill(outputVars.size)(new FloatVectorVector())
      (outputVars zip outputFVVs).foreach {
        case (vari, vect) => outputDataMap.getitem(vari).copyVariableValueToFloat(vari, vect)
      }
      outputFVVs
  }

  /** This defines and instantiates an iterator, hasNext and next are the abstract methods that
   * define the interface and inputBuffer and outputBuffer hold the input and output rows so that
   * they can be joined and returned.
   * The logic inside next checks to see if the buffer is empty, and if so sends a new batch off
   * to be evaluated.
   */
  def minibatchIterator(model: CNTKFunction,
    inputVars: Array[Variable],
    device: DeviceDescriptor,
    inputRows: Iterator[Row],
    minibatchSize: Int,
    inputColInds: Array[Int]): Iterator[Row] ={
      new Iterator[Row] {
        val inputBuffer    = new ListBuffer[Row]()
        val outputBuffer   = new ListBuffer[Row]()
        val inputShapes = inputVars.map(_.getShape)
        val inputVectorSizes = inputShapes.map(_.getTotalSize().toInt)

        def hasNext: Boolean = inputRows.hasNext || outputBuffer.nonEmpty

        def next(): Row = {
          if (outputBuffer.isEmpty) {
            inputBuffer ++= inputRows.take(minibatchSize)
            val actualMinibatchSize = inputBuffer.size
            val fvs: Array[Array[FloatVector]] =
              inputVectorSizes.map(n => Array.fill(actualMinibatchSize)(new FloatVector(n.toLong)))
            val inputFVVs: Array[FloatVectorVector] =
              Array.fill(inputVars.size)(new FloatVectorVector(actualMinibatchSize.toLong))
            inputBuffer.zipWithIndex.foreach { case (row, m) =>
              inputColInds.zipWithIndex.foreach { case (colInd, i) =>
                row.getSeq[Float](colInd).view.zipWithIndex.foreach { case (x, j) =>
                  fvs(i)(m).set(j, x)
                }
                inputFVVs(i).set(m, fvs(i)(m))
              }
            }

            val outputFVVs = applyCNTKFunction(model, inputFVVs, inputVars, device)
            assume(outputBuffer.isEmpty,
              "The output row buffer should be empty before new elements are added.")
            // outputSeqVecs has the all the output of a graph Variable in one collection,
            // therefore they must be "unzipped" to be in Rows
            // Sometimes, outputs such as cross-entropy are defined over minibatches.
            // In this case, the output is duplicated for all rows in the minibatch.
            // It is assumed that all outputs that are defined for the whole minibatch are scalars.
            // For example, if there are outputs of dimensions [1], [1], [1], and [m x n],
            // where m is the minibatch size, and n is the output vector size for one row
            // m rows will contain the same scalars repeated and the corresponding length n entries
            // of the m x n matrix

            // TODO: investigate if we want to output a batch # column so that batches can be associated
            // since spark doesn't guarantee Row ordering
            val outputSeqVecs = outputFVVs.map {
              fvv => toSeqSeq(fvv).map(fv => Vectors.dense(fv.map(_.toDouble).toArray))
            }
            val unzippedBatches =
              for (i <- 0 until actualMinibatchSize)
                yield outputSeqVecs.map(x => x(if (x.size == 1) 0 else i))
                outputBuffer ++= unzippedBatches.map(Row.fromSeq(_))
          }

          Row.merge(inputBuffer.remove(0), outputBuffer.remove(0))
        }
      }
  }

  def applyModel(inputColInds: Array[Int],
    broadcastedModel: Broadcast[CNTKFunction],
    minibatchSize: Int,
    inputNodeInds: Array[Int],
    outputNodes: Option[Array[String]])(inputRows: Iterator[Row]): Iterator[Row] = {
      val device = DeviceDescriptor.useDefaultDevice
      val m = fromSerializable(broadcastedModel.value).clone(ParameterCloningMethod.Share)
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
          minibatchIterator(model, inputVars, device, inputRows, minibatchSize, inputColInds)
    }

  private def toSeqSeq(fvv: FloatVectorVector): Seq[Seq[Float]] = {
    (0 until fvv.size.toInt)
      .map(i => (0 until fvv.get(i).size.toInt).map(j => fvv.get(i).get(j)))
  }
}

object CNTKModel extends ComplexParamsReadable[CNTKModel]

@InternalWrapper
class CNTKModel(override val uid: String) extends Model[CNTKModel]
with HasInputCol  with HasInputCols
with HasOutputCol with HasOutputCols
with ComplexParamsWritable with Wrappable {

  def this() = this(Identifiable.randomUID("CNTKModel"))

  /** Array of bytes containing the serialized CNTK <code>Function</code>
   * @group param
   */
  val model: CNTKFunctionParam =
    new CNTKFunctionParam(this, "model", "Array of bytes containing the serialized CNTKModel")

  private var broadcastedModelOption: Option[Broadcast[CNTKFunction]] = None

  /** @group setParam */
  def setModel(value: CNTKFunction): this.type = {
    // Free up memory used by the previous model
    // TODO: investigate using destroy()
    broadcastedModelOption.foreach(_.unpersist())
    broadcastedModelOption = None
    set(model, value)
  }

  /** @group getParam */
  def getModel: CNTKFunction = $(model)

  /** @group setParam */
  def setModelLocation(spark: SparkSession, path: String): this.type = {
    val modelBytes = spark.sparkContext.binaryFiles(path).first()._2.toArray
    setModel(CNTKFunction.loadModelFromBytes(modelBytes))
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

  /** Returns the dimensions of the required input */
  def getInputShape: Array[Int] =
    getModel.getInputs.get(getInputNode).getShape.getDimensions.map(_.toInt)

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
    (schema /: getOutputCols)((sch, col) => sch.add(StructField(col, VectorType)))

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  /** Evaluate the model
   * @param dataset the dataset to featurize
   * @return featurized dataset
   */
  def transform(dataset: Dataset[_]): DataFrame = {
    val spark = dataset.sparkSession
    val sc    = spark.sparkContext

    setDefault(inputCols -> getModel.getArguments.map(_.getName).toArray)
    val inputIndices = getInputCols.map(dataset.columns.indexOf(_))

    val missingCols =
      (inputIndices zip getInputCols).filter { case (ind, col) => ind == -1 }.map(_._2)
    require(missingCols.isEmpty, s"Input column(s) ${missingCols.mkString("[", ", ", "]")} do not exist")

    setDefault(inputNodes -> {
      val namedInputIndices = getInputCols.map(getModel.getArguments.indexOf)
      if (namedInputIndices.forall(_ != -1)) namedInputIndices
      else                                   Array.range(0, getInputCols.size)
    })

    val setByName  = get(outputNodeNames)
    val setByIndex = get(outputNodeIndices)

    val outputNodes: Option[Array[String]] =
      if (setByName.isDefined) setByName
      else                     setByIndex.map(_.map(getModel.getOutputs.get(_).getName))

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
            val (workDF,    outputInds, previouslyCoerced) = workDFAndOutputIndsAndPreviouslyCoerced
            val (optionUDF, inputInd)                      = optionUDFAndInputInd
            optionUDF match {
              case Some(coersionUDF) => {
                val coercedCol =
                  DatasetExtensions.produceUnusedColumnName("coerced")(workDF.columns.toSet)
                val coercedDF =
                  workDF.withColumn(coercedCol, coersionUDF(col(workDF.columns(inputInd))))
                (coercedDF, outputInds :+ workDF.columns.size, previouslyCoerced :+ coercedCol)
              }
              case None => (workDF, outputInds :+ inputIndices(outputInds.size), previouslyCoerced)
            }
          })

        val inputType           = df.schema($(inputCols).head).dataType
        val broadcastedModel = broadcastedModelOption.getOrElse(spark.sparkContext.broadcast(getModel))
        val rdd = df.rdd.mapPartitions(
          CNTKModelUtils.applyModel(selectedIndices,
            broadcastedModel,
            getMiniBatchSize,
            getInputNodes,
            outputNodes))
        if (setByName.isDefined && setByIndex.isDefined)
          throw new Exception("Must specify neither or only one of outputNodeName or outputNodeIndices")

        setDefault(outputCols -> getModel.getOutputs.map(_.getName).toArray) // defaults to all CNTK model outputs
        spark.createDataFrame(rdd, transformSchema(df.schema)).drop(coercedCols:_*)
  }

}
