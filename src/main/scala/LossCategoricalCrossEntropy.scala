package fr.adlito.nnfs
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.log

class LossCategoricalCrossEntropy extends Loss {
    override def forward(yPred: DenseMatrix[Double], yTrueOneHot: DenseMatrix[Int]): DenseVector[Double] = {
        val yTrueOneHotDouble = yTrueOneHot.map(_.toDouble)
        val mult = yPred *:* yTrueOneHotDouble
        val correctConfidences = sum(mult(*, ::))
        -log(correctConfidences)
    }

    override def forward(yPred: DenseMatrix[Double], yTrue: DenseVector[Int]): DenseVector[Double] = {
        val yTrueArray = yTrue.toArray
        val correctConfidences = DenseVector((0 until yPred.rows).map(
            i => yPred(i, yTrueArray(i))).toArray
        )
        -log(correctConfidences)
    }
}
