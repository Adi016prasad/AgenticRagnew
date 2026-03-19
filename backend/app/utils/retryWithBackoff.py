import time
import random
import logging
from functools import wraps
import asyncio

# Configure logging to see retries in your console
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=5, base_delay=1, max_delay=32, ignore_exceptions=(FileExistsError,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except ignore_exceptions as e:
                    raise e
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Max retries reached for {func.__name__}. Error: {e}")
                        raise e

                    delay = min(max_delay, base_delay * (2 ** retries))
                    sleep_time = random.uniform(0, delay)
                    
                    logger.warning(
                        f"Attempt {retries}/{max_retries} failed for {func.__name__}. "
                        f"Retrying in {sleep_time:.2f} seconds... (Error: {e})"
                    )
                    await asyncio.sleep(sleep_time)
            return None
        return wrapper
    return decorator