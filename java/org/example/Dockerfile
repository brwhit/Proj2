# Dockerfile
FROM maven:3.8-jdk-11-slim AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn package

FROM openjdk:11-jre-slim
WORKDIR /app
COPY --from=build /app/target/Predict.jar ./Predict.jar
ENTRYPOINT ["java", "-jar", "Predict.jar"]
