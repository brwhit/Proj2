FROM openjdk:8-jdk-alpine

ENV DAEMON_RUN=true
ENV SPARK_VERSION=2.4.5
ENV HADOOP_VERSION=2.7.3

WORKDIR /
COPY tmp/model/ model/
COPY Proj2.jar Proj2.jar
COPY ValidationDataset.csv ValidationDataset.csv
VOLUME /tmp

CMD ["java", "-jar", "Proj2.jar"]