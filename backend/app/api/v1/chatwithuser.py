import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging
from logging import basicConfig, getLogger, INFO
from llm.routerModel import ask_llama

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

app = FastAPI()

class sendMessageRequest(BaseModel):
    userId: str
    sessionId: str
    message: str

@app.post("/sendMessage")
async def sendMessageToUser(request : sendMessageRequest):
    logger.info(f"Sending message to user {request.userId} in session {request.sessionId}: {request.message}")
    response = ask_llama(request.message)
    logger.info(f"LLM Response: {response}")
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("createUserSession:app", host="0.0.0.0", port = 8001, reload = True)