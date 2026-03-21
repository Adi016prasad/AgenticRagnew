import json
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.services.amazonservice import uploadFileapirequest, UploadFile

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_uploader = UploadFile()

@app.post("/uploadFile")
def uploadFile(userId, fileName):
    userId = userId
    fileName = fileName
    try :
        print(s3_uploader.createDifferentObjectKeysForDifferentUsers(userId = userId, localFilePath = fileName))
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail=str(e)
        )
    except Exception as e :
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return

@app.post("/deleteFile")
def deleteFile(userId, keyName):
    userId = userId
    keyName = keyName
    try :
        s3_uploader.deleteByKeyName(userId = userId, keyName = keyName)
    except Exception as e :
        raise e

@app.post("/listTheObjects")
def listobjects(prefix):
    s3_uploader.listOfObjects(prefix)

if __name__ == "__main__":
    # uploadFile("12345", "/home/aditya/Desktop/agenticrag/backend/General FAQs (8).pdf")
    # deleteFile("12345", "12345/General FAQs (5).pdf")
    listobjects("12345/")
