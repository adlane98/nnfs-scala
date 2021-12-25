package fr.adlito.nnfs

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import breeze.stats.mean
import fr.adlito.nnfs.LayerDense
import fr.adlito.nnfs.GetData
import fr.adlito.nnfs.LossCategoricalCrossEntropy

object Test {
    def main(args: Array[String]): Unit = {
        val (x, y) = GetData.get("spiral_data", 100, 3)

        val t = System.nanoTime()
        val gt = y(::, 0).map(_.toInt)

        val dense1 = new LayerDense(2, 3)
        val relu1 = new ActivationReLU()
        val dense2 = new LayerDense(3, 3)
        val sm = new ActivationSoftmax()

        val cce = new LossCategoricalCrossEntropy()

        var lowestLoss = 999999D
        var bestDense1 = dense1.weights.copy
        var bestBias1 = dense1.bias.copy
        var bestDense2 = dense2.weights.copy
        var bestBias2 = dense2.bias.copy

        for (iter <- (0 until 10000)) {
        {
            dense1.weights += 0.05 * DenseMatrix.rand(dense1.weights.rows, dense1.weights.cols, Rand.gaussian)
            dense2.weights += 0.05 * DenseMatrix.rand(dense2.weights.rows, dense2.weights.cols, Rand.gaussian)
            dense1.bias += 0.05 * DenseVector.rand(dense1.bias.length, Rand.gaussian)
            dense2.bias += 0.05 * DenseVector.rand(dense2.bias.length, Rand.gaussian)

            dense1.forward(x)
            relu1.forward(dense1.outputs)
            dense2.forward(relu1.outputs)
            sm.forward(dense2.outputs)

            val loss = cce.calculate(sm.outputs, gt)

            val accuracy = Accuracy.calculate(sm.outputs, gt)

            if (loss < lowestLoss)
            {
                println(s"New set of weights found, iteration: ${iter}, loss: ${loss}, acc: ${accuracy}")

                bestDense1 = dense1.weights.copy
                bestBias1 = dense1.bias.copy
                bestDense2 = dense2.weights.copy
                bestBias2 = dense2.bias.copy

                lowestLoss = loss
            }
            else
            {
                dense1.weights = bestDense1.copy
                dense1.bias = bestBias1.copy
                dense2.weights = bestDense2.copy
                dense2.bias = bestBias2.copy
            }
        }

        }
        println((System.nanoTime - t) / 1e9d)


    }

    def testlayer(): Double = {
        val inputs = Array(1.2, 2.5, 3.6)
        val weights = Array(0.2, -0.5, -0.1)
        val bias = 1
        inputs.zip(weights).map(i => i._1 * i._2).sum + bias
    }

    def basicNeuronLayerWithoutBreeze(): Array[Double] = {
        val inputs = Array(1, 2, 3, 2.5)
        val weights = Array(
            Array(0.2, 0.8, -0.5, 1.0),
            Array(0.5, -0.91, 0.26, -0.5),
            Array(-0.26, -0.27, 0.17, 0.87),
        )
        val bias = Array(2, 3, 0.5)

        bias.zip(weights).map(
            layer => layer._1 + inputs.zip(layer._2).map(neuron =>
                neuron._1 * neuron._2
            ).sum
        )
    }

    def basicNeuronLayer(): DenseVector[Double] = {
        val inputs = DenseVector(1, 2, 3, 2.5)
        val weights = DenseMatrix(
            Array(0.2, 0.8, -0.5, 1.0),
            Array(0.5, -0.91, 0.26, -0.5),
            Array(-0.26, -0.27, 0.17, 0.87),
        )
        val bias = DenseVector(2, 3, 0.5)

        (inputs.t * weights.t + bias.t).t
    }

    def batchBasicNeuronLayer(): DenseMatrix[Double] = {
        val inputs = DenseMatrix(
            Array(1.0, 2.0, 3.0, 2.5),
            Array(2.0, 5.0, -1.0, 2.0),
            Array(-1.5, 2.7, 3.3, -0.8),
        )
        val weights = DenseMatrix(
            Array(0.2, 0.8, -0.5, 1.0),
            Array(0.5, -0.91, 0.26, -0.5),
            Array(-0.26, -0.27, 0.17, 0.87),
        )
        val bias = DenseVector(2, 3, 0.5)

        val mult = inputs * weights.t
        (mult.t(::, *) + bias).t
    }

    def batchMultipleLayers(): DenseMatrix[Double] = {
        val inputs = DenseMatrix(
            Array(1.0, 2.0, 3.0, 2.5),
            Array(2.0, 5.0, -1.0, 2.0),
            Array(-1.5, 2.7, 3.3, -0.8),
        )
        val weights = DenseMatrix(
            Array(0.2, 0.8, -0.5, 1.0),
            Array(0.5, -0.91, 0.26, -0.5),
            Array(-0.26, -0.27, 0.17, 0.87),
        )
        val bias = DenseVector(2, 3, 0.5)

        val weights2 = DenseMatrix(
            Array(0.1, -0.14, 0.5),
            Array(-0.5, 0.12, -0.33),
            Array(-0.44, 0.73, -0.13),
        )
        val bias2 = DenseVector(-1, 2, -0.5)

        val forward1 = inputs * weights.t
        val forwardBias1 = (forward1.t(::, *) + bias).t

        val forward2 = forwardBias1 * weights2.t
        val forwardBias2 = (forward2.t(::, *) + bias2).t

        forwardBias2
    }
}
