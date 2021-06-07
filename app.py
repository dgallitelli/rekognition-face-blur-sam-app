
import json
import logging
import os
import urllib

import boto3
import botocore
import cv2
import numpy as np

from rekognition import check_format_and_size, start_face_detection, get_timestamps_and_faces
from video_processor import apply_faces_to_video

logger = logging.getLogger()
logger.setLevel(logging.INFO)

reko = boto3.client('rekognition')
s3 = boto3.client('s3')

def lambda_handler(event, context):

    for record in event['Records']:

        # verify event has reference to S3 object
        try:
            # get metadata of file uploaded to Amazon S3
            bucket = record['s3']['bucket']['name']
            key = urllib.parse.unquote_plus(record['s3']['object']['key'])
            size = int(record['s3']['object']['size'])
            filename = key.split('/')[-1]
            local_filename = '/tmp/{}'.format(filename)
        except KeyError:
            error_message = 'Lambda invoked without S3 event data. Event needs to reference a S3 bucket and object key.'
            logger.log(error_message)
            continue

        # verify file and its size
        try:
            assert check_format_and_size(filename, size)
        except:
            error_message = 'Unsupported file type. Amazon Rekognition Video support MOV and MP4 lower than 10 GB in size'
            logger.log(error_message)
            continue

        # download file locally to /tmp retrieve metadata
        try:
            s3.download_file(bucket, key, local_filename)
        except botocore.exceptions.ClientError:
            error_message = 'Lambda role does not have permission to call GetObject for the input S3 bucket, or object does not exist.'
            logger.log(error_message)
            continue

        # use Amazon Rekognition to detect faces in image uploaded to Amazon S3
        try:
            job_id = start_face_detection(bucket, key, 1, reko)
            response = wait_for_completion(job_id, reko_client=reko)
        except rekognition.exceptions.AccessDeniedException:
            error_message = 'Lambda role does not have permission to call DetectFaces in Amazon Rekognition.'
            logger.log(error_message)
            continue
        except rekognition.exceptions.InvalidS3ObjectException:
            error_message = 'Unable to get object metadata from S3. Check object key, region and/or access permissions for input S3 bucket.'
            logger.log(error_message)
            continue

        try:
            timestamps=get_timestamps_and_faces(response, job_id, reko)
            apply_faces_to_video(timestamps, local_path_to_video, local_output, response["VideoMetadata"])
        except Exception as e:
            print(e)
            continue

        # uploaded modified image to Amazon S3 bucket
        try:
            s3.upload_file(local_output, bucket, key)
        except boto3.exceptions.S3UploadFailedError:
            error_message = 'Lambda role does not have permission to call PutObject for the output S3 bucket.'
            add_failed(bucket, error_message, failed_records, key)
            continue

        # clean up /tmp
        if os.path.exists(local_filename):
            os.remove(local_filename)

        successful_records.append({
            "bucket": bucket,
            "key": key
        })

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "cv2_version": cv2.__version__,
                "failed_records": failed_records,
                "successful_records": successful_records
            }
        )
    }
