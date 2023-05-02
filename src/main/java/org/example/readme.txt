Training Part:

1. Launch EMR 
2. Connect to main node 
3. aws s3 cp s3://project2bw/ValidationDataset.csv /home/ec2-user/tmp/ValidationDataset.csv
4. aws s3 cp s3://project2bw/TrainingDataset.csv /home/ec2-user/tmp/ValidationDataset.csv
4. java -jar train.jar


Prediction Part:
1. Saved the model in tmp folder as well
2. aws s3 cp s3://project2bw/ValidationDataset.csv /home/ec2-user/tmp/ValidationDataset.csv to get the validation dataset
3. Run the Predict.jar
4. Pass the model folder and the validation.csv to the main node
5. java -jar prediction.jar tmp/validationdataset.csv tmp/model/


https://github.com/brwhit/Proj2/upload/main/main