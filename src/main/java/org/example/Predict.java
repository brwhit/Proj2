package org.example;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.SparkConf;

public class Predict {

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

    static String[] Cols = new String[]{"fixedAcidity", "volatileAcidity", "citricAcid", "residualSugar", "chlorides", "freeSulfurDioxide",
            "totalSulfurDioxide", "density", "pH", "sulphates", "alcohol"};


    private static Dataset < Row > getValData(SparkSession spark, String test_path) {
        Dataset<Row> df_validation = spark.read()
                .format("csv")
                .schema(customSchema)
                .option("header", true)
                .option("escape", "\"")
                .option("delimiter", ";")
                .option("mode", "PERMISSIVE")
                .option("path", "tmp/ValidationDataset.csv")
                .load();

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Cols)
                .setOutputCol("features");

        //Dataset<Row> df_combined = df_train.union(df_validation);
        Dataset<Row> val_vector = assembler.transform(df_validation);
        StringIndexer indexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
        Dataset<Row> fit_val_vector = indexer.fit(val_vector).transform(val_vector);

        return fit_val_vector;
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .config("spark.driver.memory", "3g")
                .appName("Main")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        String test_path = "ValidationSet.csv";
        String model_path = "tmp\\model";

        PipelineModel pipelineModel = PipelineModel.load(model_path);
        Dataset<Row> val_df = getValData(spark, test_path).cache();
        Dataset<Row> predictions = pipelineModel.transform(val_df).cache();
        predictions.select("features", "label", "prediction").show(5, false);


        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");


        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + (1-accuracy));

        spark.stop();
    }
    }




