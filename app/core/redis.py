import json
import logging
from typing import Optional
import redis.asyncio as aioredis

logger = logging.getLogger("copilot.redis")

class RedisManager:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self, url: str = "redis://localhost:6379/0"):
        self.redis = aioredis.from_url(url, encoding="utf-8", decode_responses=True)
        await self.redis.ping()

    async def disconnect(self):
        if self.redis:
            await self.redis.close()

    async def get_cached_analysis(self, cache_key: str) -> Optional[dict]:
        if not self.redis:
            return None
        data = await self.redis.get(f"fin_cache:{cache_key}")
        if data:
            return json.loads(data)
        return None

    async def set_cached_analysis(self, cache_key: str, data: dict, ttl: int = 86400):
        if self.redis:
            await self.redis.setex(f"fin_cache:{cache_key}", ttl, json.dumps(data))

redis_manager = RedisManager()