FROM openjdk:8-jdk-alpine




ENV DAEMON_RUN=true
ENV SPARK_VERSION=2.4.5
ENV HADOOP_VERSION=2.7.3






RUN mkdir -p dataset

COPY C/Users/brwhi/IdeaProjects/untitled1/src/main/data/ data/



VOLUME /tmp

ADD target/*.jar .jar

ENTRYPOINT [ "sh", "-c", "java -Djava.security.egd=file:/dev/./urandom -jar /app.jar" ]
CMD ["data/TestDataset.csv"]