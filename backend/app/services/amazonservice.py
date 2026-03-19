import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from app.utils.retryWithBackoff import retry_with_backoff
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

MAX_S3_RETRIES = int(os.getenv("S3_MAX_RETRIES", 3))
S3_BASE_DELAY = int(os.getenv("S3_BASE_DELAY", 2))

class uploadFileapirequest:
    userId : str
    fileName : str

class UploadFile():
    def __init__(self):
        self.bucket_name = os.getenv("BUCKETNAME")
        self.region_name = os.getenv("REGION")
        self.client = boto3.client("s3", region_name = self.region_name)
    
    def check_if_exists(self, s3_key):
        try:
            self.client.head_object(Bucket = self.bucket_name, Key = s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            raise e

    @retry_with_backoff(max_retries = MAX_S3_RETRIES, base_delay = S3_BASE_DELAY)
    def createDifferentObjectKeysForDifferentUsers(self, userId, localFilePath) :
        fileName = os.path.basename(localFilePath)
        s3_folder = userId
        s3_key = f"{s3_folder.strip('/')}/{fileName}"

        try:
            with open(localFilePath, "rb") as data:
                response = self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=data,
                    ContentType="application/pdf",
                    Metadata={
                        "purpose": "testing",
                        "creator": "pdf-service-v1"
                    },
                    Tagging="environment=production&project=finance"
                )
            logger.info(response)
            logger.info("Uploaded (ETag != local MD5 or object missing)")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "PreconditionFailed":
                raise FileExistsError("File already exists in S3 with identical content")
            raise
    
    @retry_with_backoff(max_retries = MAX_S3_RETRIES, base_delay = S3_BASE_DELAY)
    def deleteByKeyName(self, userId, keyName):
        self.userId = userId
        try :
            response = self.client.delete_object(
                Bucket = self.bucket_name,
                Key = keyName
            )
            logger.info(response) # XNX7oMGrQ1ymCd136zVWlg
            logger.info("It is deleted")
        except Exception as e:
            raise e
    
    @retry_with_backoff(max_retries = MAX_S3_RETRIES, base_delay = S3_BASE_DELAY)
    def listOfObjects(self, prefix):
        prefix = prefix
        response = self.client.list_objects_v2(Bucket = self.bucket_name, Prefix = prefix)
        logger.info(response)

        # with open(localFilePath, "rb") as f:
        #     file_md5 = hashlib.md5(f.read()).hexdigest()

        #  if self.check_if_exists(s3_key):
        #      logger.warning(f"File {s3_key} already exists.")
        #      raise FileExistsError(f"The file {fileName} is already uploaded.")

        # try :
        #     with open(localFilePath, "rb") as data:
        #         self.client.put_object(
        #             Bucket = self.bucket_name,
        #             Key = s3_key,
        #             Body = data,
        #             ContentType = 'application/pdf',
        #             IfNoneMatch = "*"
        #         )
        #         logger.info(f"Successfully uploaded {s3_key}")

        #  try :
        #      uploadFile = self.client.upload_file(
        #          fileName, 
        #          self.bucket_name, 
        #          s3_key,
        #          ExtraArgs={'ContentType': 'application/pdf'}
        #      )

        #      logger.info(f"Successfully uploaded {uploadFile} {s3_key}")
        # except Exception as e:
        #     logger.info(f"Exception occured while uploading {e}")
        #     raise e