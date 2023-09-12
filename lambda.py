
#LAMBDA FUNCTION 1

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key'] #DONE
    bucket = event["s3_bucket"] #DONE
    
    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, '/tmp/image.png') ## DONE
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


#LAMBDA FUNCTION 2

import json
import base64
import subprocess
import sys
import ast


# pip install custom package to /tmp/ and add to path
subprocess.call('pip install sagemaker -t /tmp/ --no-cache-dir'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sys.path.insert(1, '/tmp/')

import sagemaker
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-09-06-09-19-25-334"

def lambda_handler(event, context):
    # Decode the image data
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)

    # For this model, the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction
    inferences = predictor.predict(image)
    
    # Convert the string representation of the list to an actual list
    inferences_list = json.loads(inferences.decode('utf-8'))

    # We return the data back to the Step Function
    event["inferences"] = inferences_list
    return {
        'statusCode': 200,
        'body': event
    }

#LAMBDA FUNCTION 3

import json

THRESHOLD = 0.8

def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event["body"]["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(value > THRESHOLD for value in inferences) #(max(inferences) > THRESHOLD)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event
    }