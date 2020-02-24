

# Define IAM role
import boto3
import re

import os
import numpy as np
import pandas as pd

import sagemaker as sage
from sagemaker import get_execution_role

role = get_execution_role()

import sagemaker as sage
from time import gmtime, strftime

sess = sage.Session()

# S3 prefix
prefix = 'sagemaker-keras-text-classification'

WORK_DIRECTORY = 'data'

data_location = sess.upload_data(WORK_DIRECTORY, key_prefix=prefix)
print(data_location) 
print(sess.default_bucket())

s3://sagemaker-us-east-1-731833107751/sagemaker-keras-text-classification
sagemaker-us-east-1-731833107751


 
bucket = sess.default_bucket()

s3 = boto3.client('s3')
s3.download_file(bucket, 'mnist-data/x_train.npy','data/x_train.npy')
s3.download_file(bucket, 'mnist-data/y_train.npy','data/y_train.npy')
s3.download_file(bucket, 'mnist-data/x_test.npy','data/x_test.npy')
s3.download_file(bucket, 'mnist-data/y_test.npy','data/y_test.npy')
 
    


account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/sagemaker-keras-text-classification:latest'.format(account, region)

tree = sage.estimator.Estimator(image,
                role, 1, 'ml.c5.2xlarge',
                output_path="s3://{}/output".format(sess.default_bucket()),
                sagemaker_session=sess)
