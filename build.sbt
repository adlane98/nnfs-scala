name := "nnfs"

version := "0.1"

scalaVersion := "2.13.7"

idePackagePrefix := Some("fr.adlito.nnfs")

libraryDependencies  ++= Seq(
    // Last stable release
    "org.scalanlp" %% "breeze" % "2.0.1-RC2",

    // The visualization library is distributed separately as well.
    // It depends on LGPL code
    "org.scalanlp" %% "breeze-viz" % "2.0.1-RC2"
)
