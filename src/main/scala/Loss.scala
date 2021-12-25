package fr.adlito.nnfs

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.pow
import breeze.stats.mean

trait Loss {
    def forward(yPred: DenseMatrix[Double], yTrue: DenseMatrix[Int]): DenseVector[Double]
    def forward(yPred: DenseMatrix[Double], yTrue: DenseVector[Int]): DenseVector[Double]

    def calculate(yPred: DenseMatrix[Double], yTrue: DenseVector[Int]): Double =
        mean(forward(yPred, yTrue))

    def clip(yPred: DenseMatrix[Double], clipValue: Double = pow(10, -7)): DenseMatrix[Double] = {
        val clipValue = pow(10, -7)
        yPred.map(
            x =>
                if (x < clipValue)
                    clipValue
                else if (x > 1 - clipValue)
                    1 - clipValue
                else x
        )
    }
}
