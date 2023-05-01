Training Part:

1. Launch EMR 
2. Connect to main node 
3. I manually put the csv's and the jars in the EC2's through WINSCP since I struggled to get the HDFS working
4. spark-submit Wine.jar


Prediction Part:
1. Saved the models in tmp folder as well
2. Run the Predict.jar
3. Pass the model and the validation.csv to the main node
4. java -jar prediction.jar test.csv model


https://github.com/brwhit/Proj2/upload/main/main