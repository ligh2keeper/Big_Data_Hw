
version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "hw_3"
  )

val sparkVersion = "3.5.0"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources()
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources())