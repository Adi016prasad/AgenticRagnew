import boto3
import os
import logging
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

# Configure logging for production visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self):
        self.bucket_name = os.getenv("BUCKETNAME")
        self.region = os.getenv("REGION")
        self.client = boto3.client("s3", region_name=self.region)
    
    def upload_single_pdf(self, local_path, s3_folder="test"):
        """
        Uploads a file to a specific S3 folder without recreating 
        the local directory hierarchy.
        """
        # 1. Extract only the filename (e.g., 'General FAQs (1).pdf')
        file_name = os.path.basename(local_path)
        
        # 2. Define the S3 Key (e.g., 'test/General FAQs (1).pdf')
        # Stripping leading slashes from folder name to avoid empty root folders
        s3_key = f"{s3_folder.strip('/')}/{file_name}"
        
        try:
            self.client.upload_file(
                local_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={'ContentType': 'application/pdf'} # Essential for PDFs
            )
            logger.info(f"Successfully uploaded to s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            logger.error(f"AWS Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")

if __name__ == "__main__":
    uploader = S3Uploader()
    
    # Absolute path from your machine
    path = "/home/aditya/Desktop/agenticrag/backend/General FAQs (1).pdf"
    
    # This will now land in 'test/General FAQs (1).pdf'
    uploader.upload_single_pdf(path, s3_folder="test")