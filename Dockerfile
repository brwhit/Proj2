# Dockerfile
# Base image
FROM openjdk:11-jdk

#FROM maven:3.5.2-jdk-8
RUN apt-get update && apt-get install -y maven

RUN apt-get update && apt-get install -y curl

# Download and extract Spark
RUN curl -O https://archive.apache.org/dist/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz && \
    tar xzf spark-3.3.1-bin-hadoop3.tgz && \
    mv spark-3.3.1-bin-hadoop3 /usr/local/spark && \
    rm spark-3.3.1-bin-hadoop3.tgz

# Download and extract Hadoop
RUN curl -O https://archive.apache.org/dist/hadoop/common/hadoop-3.3.3/hadoop-3.3.3.tar.gz && \
    tar xzf hadoop-3.3.3.tar.gz && \
    mv hadoop-3.3.3 /usr/local/hadoop && \
    rm hadoop-3.3.3.tar.gz



# Set environment variables for Hadoop and Spark
ENV HADOOP_HOME=/usr/local/hadoop
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$PATH:$HADOOP_HOME/bin:$SPARK_HOME/bin


# Set working directory
WORKDIR /app

# Copy application files


COPY target/Proj2.jar /app
COPY models /app
COPY ValidationDataset.csv /app
COPY pom.xml /app

RUN mvn clean package

# Set entrypoint
#ENTRYPOINT ["java", "-jar", "Proj2.jar"]
ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT [ "sh", "-c", "java $JAVA_OPTS -Djava.security.egd=file:/dev/./urandom -jar Proj2.jar" ]
