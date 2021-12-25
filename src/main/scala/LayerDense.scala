package fr.adlito.nnfs

import breeze.linalg._
import breeze.stats.distributions.Rand

// nFeatures = nInputs and nNeurons = nOutputs

class LayerDense(val nFeatures: Int, val nNeurons: Int){
    var weights = 0.01 * DenseMatrix.rand(nFeatures, nNeurons, Rand.gaussian)
    var bias: DenseVector[Double] = DenseVector.zeros(nNeurons)

    private var _outputs: Option[DenseMatrix[Double]] = None

    def forward(inputs: DenseMatrix[Double]): DenseMatrix[Double] = {
        val forward = inputs * weights
        _outputs = Some((forward.t(::, *) + bias).t)
        _outputs.get
    }

    def outputs: DenseMatrix[Double] = if (_outputs.isEmpty)
        throw new IllegalAccessException("Call forward before access outputs")
    else _outputs.get
}
