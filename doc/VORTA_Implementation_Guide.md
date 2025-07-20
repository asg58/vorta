# VORTA - Praktische Implementatie Gids

## 🎯 Vandaag Beginnen - Concrete Acties

### **Optie 1: GitHub Repository Setup (Aanbevolen)**

```bash
# 1. Maak nieuw repository
# Ga naar: https://github.com/new
# Repository naam: vorta-platform
# Beschrijving: "Enterprise AI Infrastructure - 5-10x more efficient than H200"
# Private repository (voor nu)

# 2. Clone lokaal
git clone https://github.com/jouw-username/vorta-platform.git
cd vorta-platform

# 3. Kopieer project structuur
# Kopieer alle .md files die we hebben gemaakt naar het nieuwe repository
```

### **Optie 2: Local Development Start**

```bash
# 1. Maak lokale directory
mkdir -p ~/vorta-platform
cd ~/vorta-platform

# 2. Initialiseer Git
git init

# 3. Maak basis structuur
mkdir -p {services,infrastructure,sdk,tools,tests,docs}
```

## 🛠️ Eerste Implementatie: Inference Engine MVP

### **1. Services Directory Setup**

```
services/inference-engine/
├── src/
│   └── vorta/
│       ├── __init__.py
│       ├── main.py                    # FastAPI app
│       ├── core/
│       │   ├── __init__.py
│       │   └── inference_engine.py    # Core inference logic
│       └── api/
│           ├── __init__.py
│           └── routes.py              # API endpoints
├── requirements.txt
├── Dockerfile
└── README.md
```

### **2. Minimale Code Implementation**

**requirements.txt:**

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
torch==2.1.0
transformers==4.35.0
faiss-cpu==1.7.4
redis==5.0.1
prometheus-client==0.19.0
httpx==0.25.2
```

**src/vorta/main.py:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import uvicorn

# Import our core modules (we'll create these next)
from .core.inference_engine import InferenceEngine
from .api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="VORTA Inference Engine",
    description="High-efficiency AI inference platform",
    version="0.1.0"
)

# Include API routes
app.include_router(router, prefix="/v1")

# Global inference engine instance
inference_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup"""
    global inference_engine
    inference_engine = InferenceEngine()
    await inference_engine.initialize()

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "vorta-inference-engine"
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "VORTA Inference Engine",
        "version": "0.1.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**src/vorta/core/inference_engine.py:**

```python
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Core VORTA inference engine with smart caching and optimization"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = {}  # Simple in-memory cache for MVP
        self.performance_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_latency": 0.0,
            "tokens_per_second": 0.0
        }

    async def initialize(self):
        """Initialize the model and tokenizer"""
        logger.info("Initializing VORTA Inference Engine...")

        try:
            # Start with a small, fast model for MVP
            model_name = "microsoft/DialoGPT-small"  # ~117MB model for testing

            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✅ Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise

    async def infer(self, prompt: str, max_tokens: int = 100, **kwargs) -> Dict[str, Any]:
        """Main inference method with caching and performance tracking"""
        start_time = time.time()

        # Check cache first (simple string-based cache for MVP)
        cache_key = f"{prompt}:{max_tokens}"
        if cache_key in self.cache:
            self.performance_stats["cache_hits"] += 1
            logger.info("🔄 Cache hit!")
            return {
                **self.cache[cache_key],
                "cached": True,
                "latency_ms": (time.time() - start_time) * 1000
            }

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.7),
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )

            # Decode response
            response_tokens = outputs[0][inputs.shape[-1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Calculate performance metrics
            end_time = time.time()
            latency = end_time - start_time
            tokens_generated = len(response_tokens)
            tokens_per_second = tokens_generated / latency if latency > 0 else 0

            # Prepare response
            result = {
                "response": response_text.strip(),
                "prompt": prompt,
                "tokens_generated": tokens_generated,
                "latency_ms": latency * 1000,
                "tokens_per_second": tokens_per_second,
                "cached": False,
                "device": self.device,
                "model_info": {
                    "name": "microsoft/DialoGPT-small",
                    "type": "causal_lm"
                }
            }

            # Cache the result
            self.cache[cache_key] = result.copy()

            # Update performance stats
            self.performance_stats["total_requests"] += 1
            self.performance_stats["avg_latency"] = (
                (self.performance_stats["avg_latency"] * (self.performance_stats["total_requests"] - 1) + latency)
                / self.performance_stats["total_requests"]
            )
            self.performance_stats["tokens_per_second"] = tokens_per_second

            logger.info(f"✅ Inference completed: {tokens_per_second:.2f} tokens/sec")
            return result

        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (
            self.performance_stats["cache_hits"] / max(self.performance_stats["total_requests"], 1) * 100
        )

        return {
            **self.performance_stats,
            "cache_hit_rate_percent": cache_hit_rate,
            "cache_size": len(self.cache),
            "device": self.device,
            "model_loaded": self.model is not None
        }
```

**src/vorta/api/routes.py:**

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request/Response models
class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_tokens: Optional[int] = Field(default=100, ge=1, le=500)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=2.0)

class InferenceResponse(BaseModel):
    response: str
    prompt: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    cached: bool
    device: str
    model_info: Dict[str, str]

class StatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    avg_latency: float
    tokens_per_second: float
    cache_hit_rate_percent: float
    cache_size: int
    device: str
    model_loaded: bool

# Dependency to get inference engine
def get_inference_engine():
    from ..main import inference_engine
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    return inference_engine

@router.post("/inference", response_model=InferenceResponse)
async def create_inference(
    request: InferenceRequest,
    engine = Depends(get_inference_engine)
):
    """Generate text using VORTA inference engine"""
    try:
        result = await engine.infer(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return InferenceResponse(**result)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.get("/stats", response_model=StatsResponse)
async def get_performance_stats(engine = Depends(get_inference_engine)):
    """Get inference engine performance statistics"""
    try:
        stats = engine.get_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

@router.post("/clear-cache")
async def clear_cache(engine = Depends(get_inference_engine)):
    """Clear the inference cache"""
    try:
        engine.cache.clear()
        return {"message": "Cache cleared successfully", "cache_size": 0}

    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")
```

### **3. Docker Setup**

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.vorta.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml (voor local development):**

```yaml
version: '3.8'

services:
  vorta-inference:
    build: .
    ports:
      - '8000:8000'
    environment:
      - PYTHONPATH=/app/src
    volumes:
      - ./src:/app/src # For development hot reload
    healthcheck:
      test: ['CMD', 'curl', '-f', 'http://localhost:8000/health']
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - '6379:6379'
    healthcheck:
      test: ['CMD', 'redis-cli', 'ping']
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - '9090:9090'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

## 🧪 Testen van de MVP

### **Lokaal Starten:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start de service
cd services/inference-engine
python -m uvicorn src.vorta.main:app --reload --host 0.0.0.0 --port 8000

# 3. Test de API
curl http://localhost:8000/health
curl http://localhost:8000/

# 4. Test inference
curl -X POST "http://localhost:8000/v1/inference" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is artificial intelligence?", "max_tokens": 50}'

# 5. Check stats
curl http://localhost:8000/v1/stats
```

### **Met Docker:**

```bash
# 1. Build image
docker build -t vorta-inference:latest .

# 2. Run container
docker run -p 8000:8000 vorta-inference:latest

# 3. Test
curl http://localhost:8000/health
```

## 📊 Verwachte Resultaten

Na deze implementatie zou je moeten hebben:

1. ✅ **Werkende API** op http://localhost:8000
2. ✅ **Interactive docs** op http://localhost:8000/docs
3. ✅ **Health check** endpoint
4. ✅ **Performance metrics** tracking
5. ✅ **Basic caching** implementatie
6. ✅ **Docker container** die start

**Deze MVP vormt de basis voor alle verdere VORTA ontwikkeling. Wil je beginnen met het implementeren van deze code?**
