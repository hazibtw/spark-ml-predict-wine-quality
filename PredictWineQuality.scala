package com.mapr.mlib

import org.apache.log4j.{Logger}
import org.apache.spark.rdd.RDD;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer

import org.apache.spark.mllib.evaluation.RegressionMetrics

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.mllib.util.MLUtils



//"fixed acidity";"volatile acidity";"citric acid";"residual sugar";
//"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";
//"pH";"sulphates";"alcohol";"quality"

//7;0.27;0.36;20.7;0.045;45;170;1.001;3;0.45;8.8;6


object PredictWineQuality {
  
   @transient lazy val logger = Logger.getLogger(getClass.getName)
  case class Wine(
      fixedAcidity:Double,
      volatileAcidity:Double,citricAcid:Double,residualSugar:Double,chlorides:Double,freeSulfurDioxide:Double,
     totalSulfurDioxide:Double,density:Double,pH:Double,sulphates:Double,alcohol:Double,
     quality:Double
  )
  
  def parseWine(line:Array[Double]):Wine={
      if(line(11)<5){        
        line(11)=1.0        
      }else if(line(11)>4 && line(11)<7){
        line(11)=2.0 
      }else{
        line(11)=3.0 
      }     
    Wine(
    line(0),
    line(1), line(2), line(3), line(4), line(5),
    line(6), line(7) , line(8), line(9) , line(10) ,
                  
line(11)
    
      )
  
}
   
   def parseRDD(rdd:RDD[String]):RDD[Array[Double]] ={
    rdd.map(_.split(";")).map(_.map(_.toDouble))
  }
   def main(args:Array[String]){
    
    val name = "Wine Predict  Application"
    logger.info(s"Starting up $name")
    
    val conf = new SparkConf().setAppName(name).setMaster("local[*]").set("spark.cores.max", "2")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    
    val sqlContext = new SQLContext(sc);
    import sqlContext._
    import sqlContext.implicits._
    
    
   val  data = sc.textFile("hdfs://localhost:8020/spark/mlib/randomforest/winequality/winequality-white.csv")
val header = data.first() 
println(header+" header");
val filterData = data.filter(row => row != header)   

    
    val wineDF = parseRDD(filterData).map(parseWine).toDF().cache()
    
    wineDF.registerTempTable("wine")
    wineDF.printSchema

    wineDF.show
    
    
    
    val featureCols =Array("fixedAcidity","volatileAcidity","citricAcid","residualSugar","chlorides","freeSulfurDioxide",
        "totalSulfurDioxide","density","pH","sulphates","alcohol")
    
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
        
        val df2=assembler.transform(wineDF);
    df2.show
    val labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label")
   
    val df3 = labelIndexer.fit(df2).transform(df2);
    df3.show
    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7,0.3),splitSeed)
    
    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(500).setFeatureSubsetStrategy("auto").setSeed(5043)
    val model = classifier.fit(trainingData);
    val evalutor = new MulticlassClassificationEvaluator().setLabelCol("label")
    val predictions = model.transform(testData);
    predictions.show(false)
    
    predictions.select("prediction","label").show(false)

   // println(model.toDebugString)
    
    val accuracy = evalutor.evaluate(predictions)
        println("accuracy before pipeline fitting" + accuracy)
   
    val wrong =predictions.select("prediction","label").map(x=> (x(0).asInstanceOf[Double]!=x(1).asInstanceOf[Double]))
    println(wrong.count());

  val accuracyManual = 1 - (wrong.count.toDouble / testData.count)
  
  println(s"accuracy model1: " + accuracyManual)


    
   val rm = new RegressionMetrics(predictions.select("prediction","label").rdd.map(x=> (x(0).asInstanceOf[Double],x(1).asInstanceOf[Double])));    
        
println("MSE: "+rm.meanSquaredError)
println("MAE: "+rm.meanAbsoluteError)
println("RMSE Squared: "+rm.rootMeanSquaredError)
println("R Squared: "+rm.r2);
println("Explained Variance: "+rm.explainedVariance +"\n");
    
val paramGrid = new ParamGridBuilder()
               .addGrid(classifier.maxBins, Array(25, 31))
               .addGrid(classifier.maxDepth, Array(5, 10))
               .addGrid(classifier.numTrees, Array(20, 60))
               .addGrid(classifier.impurity, Array("entropy","gini"))
               .build()
               
               val steps:Array[PipelineStage] = Array(classifier)
               val pipeline = new Pipeline().setStages(steps)
    val cv = new CrossValidator()
             .setEstimator(pipeline)
             .setEvaluator(evalutor)
             .setEstimatorParamMaps(paramGrid)
             .setNumFolds(10)
             
             val pipeLineFittedModel = cv.fit(trainingData)
             
             val predictations2 = pipeLineFittedModel.transform(testData)
             
              val accuracy2 = evalutor.evaluate(predictations2);

println("accuracy after pipeline fitting" + accuracy2)

println(pipeLineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))
pipeLineFittedModel.bestModel
.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0).extractParamMap

val rm2 = new RegressionMetrics(predictations2.select("prediction" , "label").rdd.map(x=> (x(0).asInstanceOf[Double],x(1).asInstanceOf[Double])))

println("MSE: "+rm2.meanSquaredError);
println("MAE: "+rm2.meanAbsoluteError);
println("RMSE Squared: "+rm2.rootMeanSquaredError)
println("R Squared: "+rm2.r2)
println("Explained Variance: "+rm2.explainedVariance +"\n");

   }
    
}