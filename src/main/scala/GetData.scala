package fr.adlito.nnfs

import java.io._

import breeze.linalg._

object GetData {
    def get(typeOfData: String, nSamples: Int, nClasses: Int, seed: Int = 0): (DenseMatrix[Double], DenseMatrix[Double]) = {
        val filenameX = s"/${typeOfData}_X_${nSamples}_${nClasses}_$seed.csv"
        val filenameY = s"/${typeOfData}_y_${nSamples}_${nClasses}_$seed.csv"

        val resourceX = getClass.getResource(filenameX)
        val resourceY = getClass.getResource(filenameY)

        (csvread(new File(resourceX.getFile)), csvread(new File(resourceY.getFile)))
    }
}
