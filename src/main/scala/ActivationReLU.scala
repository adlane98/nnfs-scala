package fr.adlito.nnfs

import breeze.linalg._

class ActivationReLU {
    private var _outputs: Option[DenseMatrix[Double]] = None

    def forward(inputs: DenseMatrix[Double]): DenseMatrix[Double] = {
        val zeros: DenseMatrix[Double] = DenseMatrix.zeros(inputs.rows, inputs.cols)
        _outputs = Some(max(zeros, inputs))
        _outputs.get
    }

    def outputs: DenseMatrix[Double] = if (_outputs.isEmpty)
        throw new IllegalAccessException("Call forward before access outputs")
    else _outputs.get
}
