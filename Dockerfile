FROM astrocrpublic.azurecr.io/runtime:3.0-2



USER root

# Install OpenJDK-17
RUN apt update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get install -y ant && \
    apt-get install -y libgomp1 && \
    apt-get clean;
RUN pip install mlflow prometheus_client    
# Set JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
RUN export JAVA_HOME

USER astro
