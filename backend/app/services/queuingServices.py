import os
import json
import logging
import asyncio
from typing import Any, Dict, Optional, Protocol
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from dotenv import load_dotenv
from app.utils.retryWithBackoff import retry_with_backoff

load_dotenv()
logger = logging.getLogger(__name__)

MAX_S3_RETRIES = int(os.getenv("KAFKA_MAX_RETRIES", 3))
S3_BASE_DELAY = int(os.getenv("KAFKA_BASE_DELAY", 2))

class MessageProducer(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, key: str, payload: Dict[str, Any]) -> Any: ...
    async def is_healthy(self) -> bool: ...

class KafkaProducerService:
    def __init__(self, bootstrap_servers: str, topic: str):
        self._topic = topic
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            retry_backoff_ms=500,
            acks='all'
        )

    async def start(self) -> None:
        try:
            await self._producer.start()
            logger.info(f"Kafka Producer started for topic: {self._topic}")
        except Exception as e:
            logger.error(f"Failed to start Kafka Producer: {e}")
            raise

    async def stop(self) -> None:
        try:
            await self._producer.stop()
            logger.info(f"Kafka Producer stopped for topic: {self._topic}")
        except Exception as e:
            logger.error(f"Error stopping Kafka Producer: {e}")

    async def is_healthy(self) -> bool:
        try:
            metadata = await self._producer.client.fetch_all_metadata()
            brokers = metadata.brokers()
            logger.info(f"Connected to {len(brokers)} brokers")
            return self._topic in metadata.topics()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    @retry_with_backoff(max_retries = MAX_S3_RETRIES, base_delay = S3_BASE_DELAY)
    async def send(self, key: str, payload: Dict[str, Any]) -> Optional[Any]:
        try:
            serialized_value = json.dumps(payload).encode('utf-8')
            metadata = await self._producer.send_and_wait(
                topic=self._topic,
                key=key.encode('utf-8'),
                value=serialized_value
            )
            logger.info(f"Message delivered to {self._topic} [p={metadata.partition}, o={metadata.offset}]")
            return metadata
        except KafkaError as e:
            logger.error(f"Kafka error during send: {e}")
            raise e

class MessagingManager:
    def __init__(self, producer: MessageProducer):
        self.producer = producer

    async def send_user_message(self, user_id: str, session_id: str, message: str, routing_key: str, **kwargs):
        payload = {
            "userId": user_id,
            "sessionId": session_id,
            "message": message,
            **kwargs
        }
        return await self.producer.send(key = routing_key, payload = payload)