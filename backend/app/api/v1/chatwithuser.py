import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.services.queuingServices import KafkaProducerService, MessagingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092,localhost:9094,localhost:9096")
KAFKA_TOPIC = os.getenv("KAFKAMESSAGEGIVINGTOPIC", "user_messages")

kafka_service : KafkaProducerService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kafka_service
    kafka_service = KafkaProducerService(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, 
        topic=KAFKA_TOPIC
    )

    await kafka_service.start()
    yield
    await kafka_service.stop()

app = FastAPI(lifespan=lifespan)

def get_messaging_manager():
    if kafka_service is None:
        raise RuntimeError("Kafka Service not initialized")

    return MessagingManager(producer = kafka_service)
class SendMessageRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

@app.post("/sendMessage")
async def send_message_to_user(request: SendMessageRequest, manager: MessagingManager = Depends(get_messaging_manager)):
    try:
        routing_key = f"{request.user_id}-{request.session_id}"
        response = await manager.send_user_message(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
            routing_key=routing_key
        )

        return {
            "status" : "Message sent",
            "metadata" : response
        }
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        return False

@app.get("/healthCheck")
async def healthCheck():
    return await kafka_service.is_healthy()

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)