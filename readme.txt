Training:

Set the credentials in the .config file, this is provided from the learner lab site

1. Launch EMR 
2. Connect to main node // Used Putty followed the instructions on AWS (https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-connect-master-node-ssh.html)

Get training and validation csv's

3. aws s3 cp s3://project2bw/ValidationDataset.csv /home/ec2-user/tmp/ValidationDataset.csv
4. aws s3 cp s3://project2bw/TrainingDataset.csv /home/ec2-user/tmp/TrainingDataset.csv

~4.5 I used WINSCP to transfer the Train and Predict.jar files since they were rather large

Distribute the datasets

5. hadoop fs -put /home/hadoop/TrainingDataset.csv /user/hadoop/TrainingDataset.csv
6. hadoop fs -put /home/hadoop/ValidationDataset.csv /user/hadoop/ValidationDataset.csv

Run the Train application and upload the saved models directory | upload models directory for convenience

7. spark-submit Train.jar
8. aws s3 cp model s3://project2bw/models --recursive  (https://www.learnaws.org/2022/03/01/aws-s3-cp-recursive/)


Prediction:

Predict.jar takes two command line arguments path/to/models and path/to/dataset
1. java -jar Predict.jar models ValidationDataset.csv


https://github.com/brwhit/Proj2
