# services/orchestrator/request_queue.py
"""
VORTA: Request Queue System

This module provides a simple, in-memory request queuing and throttling system.
Its purpose is to protect downstream services from being overwhelmed by sudden
bursts of traffic. It can queue incoming requests and process them at a
controlled rate.

Note: For a robust, distributed production system, a dedicated message queue
like RabbitMQ or Kafka would be a better choice. This is a lightweight
alternative suitable for a single-node service.
"""

import logging
import asyncio
from asyncio import Queue
from typing import Any, Callable, Awaitable, Dict

# Configure logging
logger = logging.getLogger(__name__)

class RequestProcessor:
    """
    A worker that pulls requests from a queue and processes them.
    """
    def __init__(self, queue: Queue, processing_function: Callable[[Any], Awaitable[Any]], max_concurrent: int = 5):
        self.queue = queue
        self.processing_function = processing_function
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.worker_tasks: list[asyncio.Task] = []
        logger.info(f"RequestProcessor initialized with max concurrency of {max_concurrent}.")

    async def _worker(self):
        """The core logic for a single worker."""
        while True:
            try:
                # Wait for a request to be available in the queue
                request_data, future = await self.queue.get()
                
                # Acquire the semaphore to limit concurrency
                async with self.semaphore:
                    logger.info(f"Processing request: {request_data}")
                    try:
                        result = await self.processing_function(request_data)
                        future.set_result(result)
                    except Exception as e:
                        logger.error(f"Error processing request {request_data}: {e}", exc_info=True)
                        future.set_exception(e)
                
                self.queue.task_done()
            except asyncio.CancelledError:
                logger.info("Worker task cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)

    def start(self, num_workers: int = 3):
        """Starts the specified number of worker tasks."""
        if self.worker_tasks:
            logger.warning("Workers are already running.")
            return
        
        logger.info(f"Starting {num_workers} request processing workers.")
        for _ in range(num_workers):
            task = asyncio.create_task(self._worker())
            self.worker_tasks.append(task)

    async def stop(self):
        """Stops all worker tasks gracefully."""
        logger.info("Stopping all request processing workers...")
        if not self.worker_tasks:
            return
            
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        logger.info("All workers have been stopped.")

class ThrottledQueue:
    """
    Manages the queuing of requests and their processing via workers.
    """
    def __init__(self, processing_function: Callable[[Any], Awaitable[Any]], max_queue_size: int = 100, max_concurrent_processors: int = 5):
        self.queue = Queue(maxsize=max_queue_size)
        self.processor = RequestProcessor(self.queue, processing_function, max_concurrent_processors)

    async def submit_request(self, request_data: Any) -> Any:
        """
        Submits a request to the queue and waits for its result.
        """
        if self.queue.full():
            logger.error("Request queue is full. Rejecting request.")
            raise asyncio.QueueFull("The request queue is currently full.")

        future = asyncio.get_running_loop().create_future()
        try:
            await self.queue.put((request_data, future))
            logger.info(f"Request submitted to queue. Current size: {self.queue.qsize()}")
            # Wait for the future to be resolved by the worker
            return await future
        except Exception as e:
            logger.error(f"Failed to submit request to queue: {e}")
            raise

    def start_processing(self, num_workers: int = 3):
        """Starts the background processors."""
        self.processor.start(num_workers)

    async def stop_processing(self):
        """Stops the background processors."""
        await self.processor.stop()

# --- Example Usage ---

# A dummy processing function that simulates work
async def dummy_task_processor(data: Dict[str, Any]) -> Dict[str, Any]:
    task_id = data.get("id", "unknown")
    processing_time = data.get("time", 1)
    logger.info(f"Starting processing for task {task_id} (will take {processing_time}s)...")
    await asyncio.sleep(processing_time)
    logger.info(f"Finished processing for task {task_id}.")
    return {"task_id": task_id, "status": "completed"}

async def main():
    """Demonstrates the ThrottledQueue functionality."""
    logger.info("--- Request Queue Demonstration ---")

    # 1. Initialize the queue system
    throttled_queue = ThrottledQueue(
        processing_function=dummy_task_processor,
        max_concurrent_processors=2 # Only 2 tasks can run at once
    )

    # 2. Start the background workers
    throttled_queue.start_processing(num_workers=2)

    # 3. Submit a burst of requests
    logger.info("\n--- Submitting a burst of 5 requests ---")
    tasks = []
    for i in range(5):
        request = {"id": i + 1, "time": random.randint(1, 3)}
        tasks.append(throttled_queue.submit_request(request))

    # 4. Wait for all results
    results = await asyncio.gather(*tasks)

    print("\n--- All requests processed. Results: ---")
    for res in results:
        print(res)

    # 5. Stop the queue
    await throttled_queue.stop_processing()
    logger.info("\nDemonstration complete.")

if __name__ == "__main__":
    import random
    asyncio.run(main())
