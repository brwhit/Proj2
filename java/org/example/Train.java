package org.example;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.PutObjectRequest;
import java.io.File;
import java.io.IOException;


public class Train {

    public static final StructType customSchema = new StructType(new StructField[]{
            new StructField("fixedAcidity", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("volatileAcidity", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("citricAcid", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("residualSugar", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("chlorides", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("freeSulfurDioxide", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("totalSulfurDioxide", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("density", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("pH", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("sulphates", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("alcohol", DataTypes.DoubleType, true, Metadata.empty()),
            new StructField("quality", DataTypes.DoubleType, true, Metadata.empty()),


    });

    String[] Cols = new String[]{"fixedAcidity", "volatileAcidity", "citricAcid", "residualSugar", "chlorides", "freeSulfurDioxide",
            "totalSulfurDioxide", "density", "pH", "sulphates", "alcohol"};
    String  modelPath = "./models";


    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .config("spark.driver.memory", "3g")
                .appName("Main")
                .getOrCreate();


        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        jsc.setLogLevel("ERROR");

        new Train().LogRegression_Train(spark);
        //new Train().randomForestTrain(spark);
    }

    private Dataset < Row > getTrainingData(SparkSession spark){
        Dataset<Row> df_train = spark.read()
                .format("csv")
                .schema(customSchema)
                .option("header", true)
                .option("delimiter", ";")
                .option("mode", "PERMISSIVE")
                .option("path", "./TrainingDataset.csv")
                .load();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Cols)
                .setOutputCol("features");


        Dataset<Row> train_vector = assembler.transform(df_train);
        StringIndexer indexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
        Dataset<Row> fit_train_vector = indexer.fit(train_vector).transform(train_vector);

        return fit_train_vector;
    }

    private Dataset < Row > getValData(SparkSession spark) {
        Dataset<Row> df_validation = spark.read()
                .format("csv")
                .schema(customSchema)
                .option("header", true)
                .option("escape", "\"")
                .option("delimiter", ";")
                .option("mode", "PERMISSIVE")
                .option("path", "./ValidationDataset.csv")
                .load();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Cols)
                .setOutputCol("features");

        //Dataset<Row> df_combined = df_train.union(df_validation); // tried to do cross-validation
        Dataset<Row> val_vector = assembler.transform(df_validation);
        StringIndexer indexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
        Dataset<Row> fit_val_vector = indexer.fit(val_vector).transform(val_vector);

        return fit_val_vector;
    }

    public void LogRegression_Train(SparkSession spark) throws IOException {

        Dataset < Row > input = getTrainingData(spark);
        Dataset < Row > validation = getValData(spark);
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setLabelCol("label")
                .setFeaturesCol("features");

        Pipeline pipeline1 = new Pipeline();
        pipeline1.setStages(new PipelineStage[] { lr });
        PipelineModel m1 = pipeline1.fit(input);


        System.out.println("Model saved successfully to path: " + modelPath);

        m1.write().overwrite().save("./models");

        Dataset < Row > predictions = m1.transform(validation);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + (1-accuracy));

        spark.stop();
    }


    public void randomForestTrain(SparkSession spark) throws IOException {

        Dataset < Row > input = getTrainingData(spark);
        Dataset < Row > validation = getValData(spark);


        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(10);


        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {rf});

        PipelineModel model = pipeline.fit(input);
        model.write().overwrite().save("tmp/model");

        Dataset<Row> predictions = model.transform(validation);


        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("RF Accuracy  = " + (accuracy));

        spark.stop();


    }


}