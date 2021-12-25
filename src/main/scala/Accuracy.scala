package fr.adlito.nnfs

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, sum}

object Accuracy {
    def calculate(yPred: DenseMatrix[Double], groundTruth: DenseVector[Int]): Double = {
        val predictions = argmax(yPred(*, ::))
        sum((predictions :== groundTruth)
              .toDenseVector
              .map(if (_) 1 else 0)) / predictions.length.toDouble
    }

}
