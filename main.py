"""Dipteran — mock scaffold server.

PLACEHOLDER. The streamer below drives heading with random.random() drift
instead of a real SNN, and moves the vehicle with free-form lat/lon updates
rather than the hybrid motion model (continuous heading/position, soft-pulled
toward the nearest road edge) described in PLAN.md.

Phase 2 replaces the mock step with FlyBangaloreNavigator.step() fed by real
SUMO/OSMnx state. Until then this is the running baseline.
"""

import asyncio
import math
import os
import random
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

HTML_CONTENT = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/")
def get_ui():
    return HTMLResponse(HTML_CONTENT)


@app.websocket("/ws")
async def simulation_feed(ws: WebSocket):
    await ws.accept()

    # Starting coordinates: Silk Board Junction
    lat, lon = 12.9177, 77.6238
    heading = 45.0  # Heading North-East toward Tin Factory

    events = [
        "LC4 SPIKE: Auto cut-in detected",
        "POTHOLE HIT: Campaniform sensilla shock",
        "BMTC BUS LOOM: Giant Fiber evasion triggered",
        "ZERO OPTIC FLOW: Stagnant gridlock panic",
        "JOHNSTON'S ORGAN: 110dB Pressure horn blast",
    ]

    try:
        while True:
            # Replace this mock step with FlyBangaloreNavigator.step()
            drift = (random.random() - 0.5) * 20
            heading = (heading + drift) % 360
            rad = math.radians(heading)

            # Advance coordinates based on heading
            speed = 0.00005
            lat += math.cos(rad) * speed
            lon += math.sin(rad) * speed

            # Biometrics
            pain = random.uniform(0.1, 0.9)
            panic = random.uniform(0.0, 1.0)
            active_event = random.choice(events) if random.random() < 0.15 else None

            payload = {
                "lat": lat,
                "lon": lon,
                "heading": heading,
                "sugar_reward": round(1.0 - pain, 2),
                "pain_level": round(pain, 2),
                "giant_fiber": round(panic, 2),
                "event": active_event,
            }

            await ws.send_json(payload)
            await asyncio.sleep(0.05)  # 20 Hz tick
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("DIPTERAN_HOST", "127.0.0.1"),
        port=int(os.environ.get("DIPTERAN_PORT", "8000")),
    )
