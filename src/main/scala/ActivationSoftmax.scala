package fr.adlito.nnfs

import breeze.linalg._
import breeze.numerics.exp

class ActivationSoftmax {
    private var _outputs: Option[DenseMatrix[Double]] = None

    def forward(inputs: DenseMatrix[Double]) = {
        val expValues = inputs.map(x => exp(x))
        val expSum = sum(expValues(*, ::))
        _outputs = Some(expValues(::, *) /:/ expSum)
        _outputs.get
    }

    def outputs: DenseMatrix[Double] = if (_outputs.isEmpty)
        throw new IllegalAccessException("Call forward before access outputs")
    else _outputs.get
}
