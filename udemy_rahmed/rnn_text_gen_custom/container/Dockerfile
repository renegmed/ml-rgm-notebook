# Build an image that can do training and inference in SageMaker
# See https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html

#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.14.0-cpu-py36-ubuntu16.04

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.0-cpu-py36-ubuntu18.04

RUN apt-get update && \
    apt-get install -y nginx

RUN pip install gevent gunicorn flask

ENV PATH="/opt/program:${PATH}"

# Set up the program in the imageipy
COPY code /opt/program
WORKDIR /opt/program
 
