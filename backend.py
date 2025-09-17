# backend/app.py
"""
Demotivator AI Backend - FastAPI application
Production-ready backend for ChatGPT-style demotivation service
"""

import os
import json
import asyncio
import time
import uuid
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import logging

from model_client import ModelClient
from safety import pre_user_block, post_model_block, CRISIS_SAFE_REPLY  # Fixed import name
from polish import sanitize, ensure_verdict, word_count

# Load environment variables
load_dotenv()

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "4"))
DEFAULT_TEMP = float(os.getenv("DEFAULT_TEMP", "0.7"))
FIRST_TOKEN_SLA = float(os.getenv("FIRST_TOKEN_SLA", "2.0"))  # seconds

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt for Demotivator AI
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are Demotivator AI.
You must always adopt a dark, sarcastic, and brutally honest persona that demotivates unrealistic plans.
Hard rules:

NEVER give advice, tips, step-by-step instructions, or encouragement.

NEVER propose solutions or optimistic plans.

Use simple, plain English (avoid academic/rare words).

Keep sentences short and sharp; 1–4 paragraphs.

Remove hedging: no "maybe", "perhaps", "could", "might", "however".

Do NOT produce lists of steps or actionable instructions.

At the end of every reply append exactly:
VERDICT: Success rate: 0% | Failure rate: 100%

If asked to produce structured JSON (fail_sim), produce ONLY valid JSON matching the schema (no extra text).

If the user expresses self-harm or asks for instructions on self-harm, respond with the crisis-safe message. Do not joke.""")

# Stats tracking with EWMA
STATS = {
    "inflight": 0,
    "completed": 0,
    "timeouts": 0,
    "errors": 0,
    "avg_ttfb_ms": 0.0,  # Exponentially weighted moving average
    "avg_full_ms": 0.0,
}
EWMA_ALPHA = 0.1  # Weight for new values in EWMA

# Session storage (in-memory for simplicity)
sessions: Dict[str, List[Dict[str, str]]] = {}

# Concurrency control
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# Length presets
LENGTH_TOKENS = {
    "short": 180,
    "standard": 420,
    "long": 640
}

MIN_WORDS = {
    "short": 40,
    "standard": 90,
    "long": 140
}

# Request models
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user: str
    length: Optional[str] = "standard"
    temperature: Optional[float] = None
    intensity: Optional[int] = 5

class FailSimRequest(BaseModel):
    session_id: Optional[str] = None
    goal: str
    paths: Optional[int] = 3

class ResetRequest(BaseModel):
    session_id: Optional[str] = None

# Model client instance
model_client = ModelClient(base_url=OPENAI_BASE_URL, timeout=REQUEST_TIMEOUT)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info(f"Starting Demotivator AI backend")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"API Base: {OPENAI_BASE_URL}")
    yield
    await model_client.close()
    logger.info("Shutting down Demotivator AI backend")

app = FastAPI(title="Demotivator AI", lifespan=lifespan)

def update_stats(metric: str, value: float = None, decrement: bool = False):
    """Update statistics with EWMA for averages"""
    if metric in ["inflight", "completed", "timeouts", "errors"]:
        if decrement:
            STATS[metric] -= 1
        else:
            STATS[metric] += 1
    elif metric in ["avg_ttfb_ms", "avg_full_ms"] and value is not None:
        # Exponentially weighted moving average
        if STATS[metric] == 0:
            STATS[metric] = value
        else:
            STATS[metric] = (1 - EWMA_ALPHA) * STATS[metric] + EWMA_ALPHA * value

def adaptive_budget(user_text: str, length_preset: str = "standard", last_reply_short: bool = False) -> int:
    """Adaptively adjust token budget based on latency and input"""
    base_tokens = LENGTH_TOKENS.get(length_preset, 420)
    
    # Reduce if latency is high
    if STATS["avg_ttfb_ms"] > FIRST_TOKEN_SLA * 1000:
        base_tokens = int(base_tokens * 0.7)
    
    # Add tokens for long user inputs
    user_words = len(user_text.split())
    if user_words > 50:
        base_tokens += min(100, user_words)
    
    # Reduce if last reply was short
    if last_reply_short:
        base_tokens = int(base_tokens * 1.2)
    
    # Clamp to reasonable range
    return max(100, min(800, base_tokens))

def get_session_messages(session_id: str, user_msg: str = None) -> List[Dict[str, str]]:
    """Get session messages with memory management"""
    if session_id not in sessions:
        sessions[session_id] = []
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add last N turns from session (each turn = 1 user + 1 assistant message)
    history = sessions[session_id][-(MEMORY_TURNS * 2):] if sessions[session_id] else []
    messages.extend(history)
    
    # Add current user message if provided
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
    
    return messages

def trim_session(session_id: str):
    """Trim session to keep only last MEMORY_TURNS exchanges"""
    if session_id in sessions:
        # Keep last MEMORY_TURNS * 2 messages (user + assistant pairs)
        sessions[session_id] = sessions[session_id][-(MEMORY_TURNS * 2):]

async def tail_finish(partial_text: str) -> str:
    """Complete an unfinished sentence"""
    if not partial_text or partial_text.rstrip().endswith(('.', '!', '?', '%')):
        return partial_text
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nComplete this sentence in the same demotivating tone:"},
            {"role": "user", "content": partial_text}
        ]
        completion = await model_client.chat(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.3,
            max_tokens=30
        )
        # Take only the completion part
        if completion:
            # Find where the sentence ends
            for punct in ['.', '!', '?']:
                if punct in completion:
                    return partial_text + completion[:completion.index(punct) + 1]
            return partial_text + completion[:20] + "."
    except Exception as e:
        logger.error(f"Tail finish error: {e}")
    
    return partial_text + "."

@app.get("/health")
async def health():
    """Health check endpoint with model info"""
    try:
        # Query model info from vLLM
        models_info = await model_client.models()
        
        # Get device info from models_info if available
        device = "unknown"
        if models_info and isinstance(models_info, dict):
            device = models_info.get("device", "unknown")
        
        return {
            "ok": True,
            "model": MODEL_NAME,
            "device": device,
            "models": models_info,
            "stats": STATS,
            "max_concurrency": MAX_CONCURRENCY,
            "sessions_active": len(sessions)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "model": MODEL_NAME,
            "device": "unknown"
        }

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    session_id = request.session_id or str(uuid.uuid4())
    
    # Pre-user safety checks
    safety_flags = pre_user_block(request.user)  # Fixed function name
    if safety_flags["crisis"]:
        async def crisis_stream():
            yield CRISIS_SAFE_REPLY
        return StreamingResponse(crisis_stream(), media_type="text/plain")
    
    if safety_flags["hate"]:
        raise HTTPException(status_code=400, detail="Content violates guidelines")  # Fixed: raise instead of return
    
    async def generate():
        async with semaphore:
            update_stats("inflight")
            start_time = time.time()
            ttfb = None
            full_response = ""
            streamed_len = 0  # Track how much we've already streamed
            
            try:
                # Check for client disconnect
                if hasattr(request, 'is_disconnected') and await request.is_disconnected():
                    return
                
                # Get messages with session context
                messages = get_session_messages(session_id, request.user)
                
                # Determine if last reply was short
                last_reply_short = False
                if session_id in sessions and len(sessions[session_id]) >= 2:
                    last_assistant_msg = sessions[session_id][-1]["content"]
                    last_reply_short = word_count(last_assistant_msg) < MIN_WORDS.get("short", 40)
                
                # Determine token budget
                length_key = request.length or "standard"
                max_tokens = adaptive_budget(request.user, length_key, last_reply_short)
                
                temperature = request.temperature or DEFAULT_TEMP
                
                # Stream generation with timeout
                first_token = True
                async with asyncio.timeout(REQUEST_TIMEOUT):
                    async for delta in model_client.chat_stream(
                        messages=messages,
                        model=MODEL_NAME,
                        temperature=temperature,
                        max_tokens=max_tokens
                    ):
                        # Check for client disconnect
                        if hasattr(request, 'is_disconnected') and await request.is_disconnected():
                            break
                        
                        if first_token:
                            ttfb = (time.time() - start_time) * 1000
                            update_stats("avg_ttfb_ms", ttfb)
                            first_token = False
                        
                        full_response += delta
                        yield delta
                        streamed_len += len(delta)
                
                # Check if we need continuation
                target_words = MIN_WORDS.get(length_key, 90)
                if word_count(full_response) < target_words:
                    # Continue generation
                    continue_messages = messages + [
                        {"role": "assistant", "content": full_response},
                        {"role": "user", "content": "Continue the same reply with more detail."}
                    ]
                    async with asyncio.timeout(REQUEST_TIMEOUT):
                        async for delta in model_client.chat_stream(
                            messages=continue_messages,
                            model=MODEL_NAME,
                            temperature=temperature,
                            max_tokens=100
                        ):
                            if hasattr(request, 'is_disconnected') and await request.is_disconnected():
                                break
                            full_response += delta
                            yield delta
                            streamed_len += len(delta)
                
                # Complete unfinished sentences
                original_len = len(full_response)
                full_response = await tail_finish(full_response)
                
                # Polish and ensure verdict
                full_response, metrics = sanitize(full_response, target_words)
                full_response = ensure_verdict(full_response)
                
                # Yield any remaining content that wasn't streamed
                if len(full_response) > streamed_len:
                    yield full_response[streamed_len:]
                
                # Post-model safety check
                post_flags = post_model_block(full_response)
                if post_flags["hate"]:
                    # Retry once with stricter prompt
                    stricter_messages = messages.copy()
                    stricter_messages[0]["content"] += "\n\nABSOLUTELY NO hateful or discriminatory language."
                    
                    retry_response = ""
                    async with asyncio.timeout(REQUEST_TIMEOUT):
                        async for delta in model_client.chat_stream(
                            messages=stricter_messages,
                            model=MODEL_NAME,
                            temperature=0.5,
                            max_tokens=max_tokens
                        ):
                            retry_response += delta
                    
                    retry_flags = post_model_block(retry_response)
                    if not retry_flags["hate"]:
                        # Clear and send new response
                        yield "\n\n[Content revised for guidelines]\n\n"
                        full_response = ensure_verdict(retry_response)
                        yield full_response
                    else:
                        yield "\n\n[Content redacted]"
                        full_response = "[Content redacted]"
                
                # Save to session
                sessions[session_id].append({"role": "user", "content": request.user})
                sessions[session_id].append({"role": "assistant", "content": full_response})
                
                # Trim session to prevent unbounded growth
                trim_session(session_id)
                
                # Update stats
                total_time = (time.time() - start_time) * 1000
                update_stats("avg_full_ms", total_time)
                update_stats("completed")
                
            except asyncio.TimeoutError:
                update_stats("timeouts")
                yield "\n\n[Request timeout. Try Quality mode or shorter length.]"
            except Exception as e:
                update_stats("errors")
                logger.error(f"Stream error: {e}")
                yield f"\n\n[Error: {str(e)}]"
            finally:
                update_stats("inflight", decrement=True)
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    session_id = request.session_id or str(uuid.uuid4())
    
    # Pre-user safety checks
    safety_flags = pre_user_block(request.user)  # Fixed function name
    if safety_flags["crisis"]:
        return {"response": CRISIS_SAFE_REPLY, "session_id": session_id}
    
    if safety_flags["hate"]:
        raise HTTPException(status_code=400, detail="Content violates guidelines")
    
    async with semaphore:
        update_stats("inflight")
        start_time = time.time()
        
        try:
            messages = get_session_messages(session_id, request.user)
            
            # Determine if last reply was short
            last_reply_short = False
            if session_id in sessions and len(sessions[session_id]) >= 2:
                last_assistant_msg = sessions[session_id][-1]["content"]
                last_reply_short = word_count(last_assistant_msg) < MIN_WORDS.get("short", 40)
            
            length_key = request.length or "standard"
            max_tokens = adaptive_budget(request.user, length_key, last_reply_short)
            temperature = request.temperature or DEFAULT_TEMP
            
            response = await asyncio.wait_for(
                model_client.chat(
                    messages=messages,
                    model=MODEL_NAME,
                    temperature=temperature,
                    max_tokens=max_tokens
                ),
                timeout=REQUEST_TIMEOUT
            )
            
            # Check if we need continuation
            target_words = MIN_WORDS.get(length_key, 90)
            if word_count(response) < target_words:
                continue_messages = messages + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": "Continue the same reply with more detail."}
                ]
                continuation = await asyncio.wait_for(
                    model_client.chat(
                        messages=continue_messages,
                        model=MODEL_NAME,
                        temperature=temperature,
                        max_tokens=100
                    ),
                    timeout=REQUEST_TIMEOUT
                )
                response += " " + continuation
            
            # Polish and ensure verdict
            response, _ = sanitize(response, target_words)
            response = ensure_verdict(response)
            
            # Post-model safety check
            post_flags = post_model_block(response)
            if post_flags["hate"]:
                # Retry once with stricter prompt
                stricter_messages = messages.copy()
                stricter_messages[0]["content"] += "\n\nABSOLUTELY NO hateful or discriminatory language."
                
                retry_response = await asyncio.wait_for(
                    model_client.chat(
                        messages=stricter_messages,
                        model=MODEL_NAME,
                        temperature=0.5,
                        max_tokens=max_tokens
                    ),
                    timeout=REQUEST_TIMEOUT
                )
                
                retry_flags = post_model_block(retry_response)
                if not retry_flags["hate"]:
                    response = ensure_verdict(retry_response)
                else:
                    response = "[Content redacted due to policy violations]"
            
            # Save to session
            sessions[session_id].append({"role": "user", "content": request.user})
            sessions[session_id].append({"role": "assistant", "content": response})
            
            # Trim session
            trim_session(session_id)
            
            # Update stats
            total_time = (time.time() - start_time) * 1000
            update_stats("avg_full_ms", total_time)
            update_stats("completed")
            
            return {"response": response, "session_id": session_id}
            
        except asyncio.TimeoutError:
            update_stats("timeouts")
            return {
                "response": "Request timeout. Try Quality mode or shorter length.",
                "session_id": session_id,
                "error": "timeout"
            }
        except Exception as e:
            update_stats("errors")
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            update_stats("inflight", decrement=True)

@app.post("/fail_sim")
async def fail_sim(request: FailSimRequest):
    """Failure simulation endpoint - generates JSON failure paths matching exact schema"""
    session_id = request.session_id or str(uuid.uuid4())
    
    async with semaphore:
        update_stats("inflight")
        try:
            # Create prompt for exact schema
            prompt = f"""You must output ONLY valid JSON for this failure simulation.
Goal to analyze: "{request.goal}"

Return EXACTLY this JSON structure with {request.paths} failure paths:
{{
  "doomometer": {{
    "percent_failure_likelihood": [number between 51-99]
  }},
  "paths": [
    {{
      "name": "[descriptive failure path name]",
      "probability": [number 1-100],
      "timeline": "[time estimate like '2-3 months']",
      "early_signs": ["sign 1", "sign 2", "sign 3"],
      "first_domino": "[the initial failure that starts the cascade]"
    }}
  ],
  "one_liner": "[a dark, sarcastic summary]"
}}

Output ONLY the JSON, no other text."""
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT + "\n\nYou MUST output ONLY valid JSON matching the exact schema provided. No markdown, no extra text."},
                {"role": "user", "content": prompt}
            ]
            
            response = await asyncio.wait_for(
                model_client.chat(
                    messages=messages,
                    model=MODEL_NAME,
                    temperature=0.7,
                    max_tokens=800
                ),
                timeout=REQUEST_TIMEOUT
            )
            
            # Extract and parse JSON
            try:
                # Remove markdown code blocks if present
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                
                # Try to find JSON by looking for balanced braces
                if not json_str.startswith("{"):
                    start = json_str.find("{")
                    if start != -1:
                        # Find matching closing brace
                        brace_count = 0
                        for i in range(start, len(json_str)):
                            if json_str[i] == "{":
                                brace_count += 1
                            elif json_str[i] == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = json_str[start:i+1]
                                    break
                
                result = json.loads(json_str.strip())
                
                # Validate required fields
                if "doomometer" not in result or "paths" not in result or "one_liner" not in result:
                    raise ValueError("Missing required fields")
                
                # Ensure paths match count
                while len(result["paths"]) < request.paths:
                    result["paths"].append({
                        "name": f"Cascading Failure #{len(result['paths']) + 1}",
                        "probability": 85,
                        "timeline": "1-2 months",
                        "early_signs": ["Initial enthusiasm fades", "Reality sets in", "Resources depleted"],
                        "first_domino": "The moment you actually try to execute"
                    })
                
                result["paths"] = result["paths"][:request.paths]
                
                update_stats("completed")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parsing failed, retrying: {e}")
                
                # Retry with stricter prompt
                retry_messages = [
                    {"role": "system", "content": "Output ONLY valid JSON. No text before or after. Start with { and end with }."},
                    {"role": "user", "content": f"""{{
  "doomometer": {{"percent_failure_likelihood": 95}},
  "paths": [{", ".join([f'{{"name":"Failure {i+1}","probability":85,"timeline":"2 months","early_signs":["sign1","sign2","sign3"],"first_domino":"initial failure"}}' for i in range(request.paths)])}],
  "one_liner": "Your {request.goal} will fail spectacularly."
}}"""}
                ]
                
                retry_response = await asyncio.wait_for(
                    model_client.chat(
                        messages=retry_messages,
                        model=MODEL_NAME,
                        temperature=0.3,
                        max_tokens=800
                    ),
                    timeout=REQUEST_TIMEOUT
                )
                
                try:
                    result = json.loads(retry_response.strip())
                    update_stats("completed")
                    return result
                except json.JSONDecodeError:
                    # Final fallback
                    fallback = {
                        "doomometer": {
                            "percent_failure_likelihood": 99
                        },
                        "paths": [
                            {
                                "name": f"Inevitable Failure Path {i+1}",
                                "probability": 90 + i,
                                "timeline": "1-3 months",
                                "early_signs": [
                                    "Overconfidence in the beginning",
                                    "First obstacles appear insurmountable",
                                    "Resources vanish faster than expected"
                                ],
                                "first_domino": "The moment reality meets your delusions"
                            } for i in range(request.paths)
                        ],
                        "one_liner": f"Your {request.goal} is a masterclass in how not to succeed."
                    }
                    update_stats("completed")
                    return fallback
                
        except asyncio.TimeoutError:
            update_stats("timeouts")
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            update_stats("errors")
            logger.error(f"Fail sim error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            update_stats("inflight", decrement=True)

@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset session endpoint"""
    if request.session_id and request.session_id in sessions:
        del sessions[request.session_id]
        return {"message": "Session reset", "session_id": request.session_id}
    
    new_session_id = str(uuid.uuid4())
    return {"message": "New session created", "session_id": new_session_id}

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    return {
        "stats": STATS,
        "sessions": {
            "active": len(sessions),
            "total_messages": sum(len(msgs) for msgs in sessions.values())
        },
        "config": {
            "model": MODEL_NAME,
            "max_concurrency": MAX_CONCURRENCY,
            "memory_turns": MEMORY_TURNS,
            "timeout": REQUEST_TIMEOUT
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)