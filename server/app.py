"""FastAPI server for the Hospital Scheduler OpenEnv environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server.http_server import create_app
from models import HospitalAction, HospitalObservation, ActionType
from server.hospital_environment import HospitalEnvironment, TASK_IDS

app = create_app(
    HospitalEnvironment,
    HospitalAction,
    HospitalObservation,
    env_name="hospital_scheduler",
    max_concurrent_envs=4,
)

import asyncio
import re
import os
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Allow file:// origin and any localhost port (dev only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


class TaskInfo(BaseModel):
    task_id: str
    difficulty: str
    prompt: str
    max_steps: int
    target_patients: list[str]


class ActionSchema(BaseModel):
    action_type: str
    description: str
    required_params: list[str]
    optional_params: list[str]


ACTION_SCHEMAS = [
    ActionSchema(action_type="search_doctors", description="Find doctors by department and date", required_params=["department"], optional_params=["date"]),
    ActionSchema(action_type="check_availability", description="Check a doctor's open time slots", required_params=["doctor_id"], optional_params=["date"]),
    ActionSchema(action_type="book_appointment", description="Book patient into a time slot", required_params=["doctor_id", "patient_id", "slot_id"], optional_params=["urgency"]),
    ActionSchema(action_type="cancel_appointment", description="Cancel an existing appointment", required_params=["appointment_id"], optional_params=[]),
    ActionSchema(action_type="reschedule_appointment", description="Move appointment to new slot", required_params=["appointment_id", "new_slot_id"], optional_params=[]),
    ActionSchema(action_type="get_patient_info", description="Get patient details, symptoms, history, allergies", required_params=["patient_id"], optional_params=[]),
    ActionSchema(action_type="list_departments", description="List all hospital departments", required_params=[], optional_params=[]),
    ActionSchema(action_type="get_appointment_details", description="Look up an appointment", required_params=["appointment_id"], optional_params=[]),
    ActionSchema(action_type="verify_insurance", description="Check if patient's insurance covers a department", required_params=["patient_id", "department"], optional_params=[]),
    ActionSchema(action_type="check_waitlist", description="View waitlist for a department", required_params=[], optional_params=["department", "patient_id"]),
    ActionSchema(action_type="add_to_waitlist", description="Add patient to department waitlist", required_params=["patient_id", "department"], optional_params=["preferred_doctor_id"]),
    ActionSchema(action_type="get_doctor_schedule", description="Get full doctor schedule", required_params=["doctor_id"], optional_params=[]),
    ActionSchema(action_type="get_working_hours", description="Get doctor's shift schedule and working days", required_params=["doctor_id"], optional_params=[]),
    ActionSchema(action_type="request_referral", description="Get a GP referral for a patient before specialist booking", required_params=["patient_id", "referring_doctor_id"], optional_params=[]),
    ActionSchema(action_type="request_preauth", description="Request insurance pre-authorization for a department", required_params=["patient_id", "department"], optional_params=[]),
    ActionSchema(action_type="check_preauth_status", description="Check pre-authorization status for a patient and department", required_params=["patient_id", "department"], optional_params=[]),
    ActionSchema(action_type="check_waitlist_offers", description="Check for slot offers triggered by cancellations", required_params=[], optional_params=["patient_id"]),
    ActionSchema(action_type="accept_waitlist_offer", description="Accept a waitlist slot offer and book the appointment", required_params=["patient_id"], optional_params=[]),
    ActionSchema(action_type="finish", description="Signal task completion -- call when all goals are met", required_params=[], optional_params=[]),
]


@app.get("/tasks", tags=["Environment Info"])
def list_tasks():
    """Return list of tasks with action schema."""
    tasks_dir = Path(__file__).resolve().parent.parent / "tasks"
    tasks = []
    for tid in TASK_IDS:
        path = tasks_dir / f"{tid}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            tasks.append(TaskInfo(
                task_id=data["task_id"],
                difficulty=data["difficulty"],
                prompt=data["prompt"],
                max_steps=data["max_steps"],
                target_patients=data.get("target_patients", []),
            ))
    return {
        "tasks": [t.model_dump() for t in tasks],
        "action_schema": [a.model_dump() for a in ACTION_SCHEMAS],
    }


@app.get("/app", tags=["Frontend"], include_in_schema=False)
def serve_frontend():
    """Serve the hospital scheduler frontend UI."""
    html_path = _FRONTEND_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"error": "Frontend not found"}


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint — required by hackathon validator."""
    return {"status": "ok", "name": "hospital_scheduler", "version": "2.1.0"}


@app.get("/grader", tags=["Environment Info"])
def run_grader():
    """Return grader score after an episode is completed.

    Note: This uses a fresh environment instance since the openenv
    server manages environments via WebSocket sessions. For proper
    grading, use the /baseline endpoint or run baseline.py directly.
    """
    # The openenv framework manages env instances internally.
    # This endpoint creates a temporary env for demonstration.
    return {
        "message": "Use /baseline to run full episodes with grading, or run baseline.py directly.",
        "available_tasks": TASK_IDS,
    }


@app.get("/baseline", tags=["Environment Info"])
def run_baseline():
    """Trigger the heuristic baseline and return scores for all tasks."""
    from env import HospitalEnv
    from graders import grade as grade_fn
    from models import ActionType as AT

    # Import heuristic sequences
    HEURISTIC_SEQUENCES = _get_heuristic_sequences()

    results = {}
    for task_id in TASK_IDS:
        env = HospitalEnv()
        env.reset(task_id)

        steps = HEURISTIC_SEQUENCES.get(task_id, [])
        for atype, params in steps:
            from models import ActionType as AT2
            internal_action_module = __import__("env", fromlist=["Action"])
            Action = getattr(internal_action_module, "Action")
            action = Action(action_type=AT2(atype), parameters=params)
            obs = env.step(action)
            if obs.done:
                break

        score = grade_fn(env)
        results[task_id] = {
            "score": round(score, 2),
            "steps_used": env.step_number,
            "max_steps": env.task_config.max_steps if env.task_config else 0,
        }

    avg = sum(r["score"] for r in results.values()) / len(results) if results else 0
    return {
        "agent": "heuristic",
        "results": results,
        "average_score": round(avg, 2),
    }


def _get_heuristic_sequences():
    """Return heuristic action sequences for all tasks."""
    return {
        "task_easy": [
            ("get_patient_info", {"patient_id": "P001"}),
            ("search_doctors", {"department": "cardiology", "date": "2026-04-01"}),
            ("check_availability", {"doctor_id": "D001", "date": "2026-04-01"}),
            # D001-0401-09 is pre-booked by APT-202; use next available slot
            ("book_appointment", {"doctor_id": "D001", "patient_id": "P001", "slot_id": "D001-0401-10"}),
        ],
        "task_medium": [
            ("get_patient_info", {"patient_id": "P002"}),
            ("get_appointment_details", {"appointment_id": "APT-101"}),
            ("check_availability", {"doctor_id": "D003", "date": "2026-04-06"}),
            ("reschedule_appointment", {"appointment_id": "APT-101", "new_slot_id": "D003-0406-14"}),
        ],
        "task_hard": [
            ("get_patient_info", {"patient_id": "P003"}),
            ("search_doctors", {"department": "neurology", "date": "2026-04-01"}),
            ("check_availability", {"doctor_id": "D007", "date": "2026-04-01"}),
            # D007-0401-09 is pre-booked by APT-505; use next available slot
            ("book_appointment", {"doctor_id": "D007", "patient_id": "P003", "slot_id": "D007-0401-14"}),
            ("get_patient_info", {"patient_id": "P004"}),
            ("search_doctors", {"department": "orthopedics", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D003", "patient_id": "P004", "slot_id": "D003-0401-14"}),
            ("get_patient_info", {"patient_id": "P005"}),
            ("get_appointment_details", {"appointment_id": "APT-202"}),
            ("search_doctors", {"department": "cardiology", "date": "2026-04-02"}),
            ("book_appointment", {"doctor_id": "D001", "patient_id": "P005", "slot_id": "D001-0402-09"}),
        ],
        "task_expert": [
            ("get_patient_info", {"patient_id": "P002"}),
            ("verify_insurance", {"patient_id": "P002", "department": "orthopedics"}),
            ("search_doctors", {"department": "general_medicine", "date": "2026-04-01"}),
            ("request_referral", {"patient_id": "P002", "referring_doctor_id": "D009"}),
            ("request_preauth", {"patient_id": "P002", "department": "orthopedics"}),
            ("search_doctors", {"department": "orthopedics", "date": "2026-04-01"}),
            ("check_availability", {"doctor_id": "D003", "date": "2026-04-01"}),
            # D003-0401-09 is blocked by APT-101; use next available slot
            ("book_appointment", {"doctor_id": "D003", "patient_id": "P002", "slot_id": "D003-0401-14"}),
            ("get_patient_info", {"patient_id": "P009"}),
            ("get_appointment_details", {"appointment_id": "APT-404"}),
            ("verify_insurance", {"patient_id": "P009", "department": "general_medicine"}),
            ("search_doctors", {"department": "general_medicine", "date": "2026-04-02"}),
            ("book_appointment", {"doctor_id": "D009", "patient_id": "P009", "slot_id": "D009-0402-10"}),
        ],
        "task_nightmare": [
            ("get_patient_info", {"patient_id": "P008"}),
            ("verify_insurance", {"patient_id": "P008", "department": "cardiology"}),
            ("get_appointment_details", {"appointment_id": "APT-303"}),
            ("search_doctors", {"department": "cardiology", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D001", "patient_id": "P008", "slot_id": "D001-0401-10", "urgency": "emergency"}),
            ("get_patient_info", {"patient_id": "P007"}),
            ("verify_insurance", {"patient_id": "P007", "department": "pediatrics"}),
            ("search_doctors", {"department": "pediatrics", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D012", "patient_id": "P007", "slot_id": "D012-0401-11", "urgency": "urgent"}),
            ("get_patient_info", {"patient_id": "P010"}),
            ("check_waitlist", {"department": "neurology"}),
            # Cancel APT-505 (neurology slot) to trigger waitlist notification for P010
            ("cancel_appointment", {"appointment_id": "APT-505"}),
            ("check_waitlist_offers", {"patient_id": "P010"}),
            ("verify_insurance", {"patient_id": "P010", "department": "neurology"}),
            ("accept_waitlist_offer", {"patient_id": "P010"}),
            ("get_patient_info", {"patient_id": "P006"}),
            ("verify_insurance", {"patient_id": "P006", "department": "dermatology"}),
            ("search_doctors", {"department": "dermatology", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D005", "patient_id": "P006", "slot_id": "D005-0401-09"}),
        ],
    }


AGENT_SYSTEM_PROMPT = """You are a hospital appointment scheduling agent at CrestView Medical Center (v2.1.0).
You interact with a hospital management system by choosing one action per turn.

Available actions (use exact action_type values):
- get_patient_info        {"patient_id": "P001"}
- search_doctors          {"department": "cardiology", "date": "2026-04-01"}
- check_availability      {"doctor_id": "D001", "date": "2026-04-01"}
- book_appointment        {"doctor_id": "D001", "patient_id": "P001", "slot_id": "D001-0401-09", "appointment_type": "consultation"}
- reschedule_appointment  {"appointment_id": "APT-101", "new_slot_id": "D003-0406-14"}
- cancel_appointment      {"appointment_id": "APT-101"}
- get_appointment_details {"appointment_id": "APT-101"}
- verify_insurance        {"patient_id": "P001", "department": "cardiology"}
- list_departments        {}
- check_waitlist          {"department": "neurology"}
- add_to_waitlist         {"patient_id": "P001", "department": "neurology"}
- get_doctor_schedule     {"doctor_id": "D001"}
- get_working_hours       {"doctor_id": "D001"}
- request_referral        {"patient_id": "P001", "referring_doctor_id": "D009"}
- request_preauth         {"patient_id": "P001", "department": "orthopedics"}
- check_preauth_status    {"patient_id": "P001", "department": "orthopedics"}
- check_waitlist_offers   {"patient_id": "P001"}
- accept_waitlist_offer   {"patient_id": "P001"}
- finish                  {}

Appointment types: follow_up (15 min), consultation (30 min), procedure (60 min).
Doctors have working_hours and working_days — only book within their schedule.

Rules:
1. Call get_patient_info first.
2. Call verify_insurance before booking — some plans require referral or pre-authorization.
3. If referral required: call request_referral with a GP doctor_id before booking.
4. If preauth required: call request_preauth then check_preauth_status (must be approved).
5. Always call check_availability before booking.
6. Book patients in triage order: emergency > urgent > routine.
7. Call finish when all goals are met.
8. Never repeat a failed action — adapt based on the error.

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"action_type": "...", "parameters": {...}}"""


class AgentRunRequest(BaseModel):
    task_id: str
    api_base: str = "https://router.huggingface.co/v1"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    api_key: Optional[str] = None
    max_steps: Optional[int] = None


def _parse_action(text: str) -> dict | None:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/agent/run", tags=["Agent"])
async def agent_run(req: AgentRunRequest):
    """Run an LLM agent on a task and stream steps as Server-Sent Events."""
    from env import HospitalEnv, Action
    from graders import grade as grade_fn
    from models import ActionType as AT

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai package not installed")

    api_key = req.api_key or os.environ.get("HF_TOKEN", "noop")
    client = OpenAI(base_url=req.api_base, api_key=api_key)

    async def stream():
        env = HospitalEnv()
        obs = env.reset(req.task_id)
        max_steps = req.max_steps or obs.max_steps or 30

        yield _sse("reset", {
            "task_id": req.task_id,
            "message": obs.message,
            "max_steps": max_steps,
        })

        messages = [
            {"role": "user", "content": f"Task:\n{obs.message}\n\nBegin. What is your first action?"}
        ]

        for step_i in range(1, max_steps + 1):
            # Ask LLM
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=req.model_name,
                    messages=[{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                llm_text = response.choices[0].message.content or ""
            except Exception as e:
                yield _sse("error", {"message": f"LLM error: {e}"})
                break

            action_data = _parse_action(llm_text)
            if action_data is None:
                yield _sse("thinking", {"step": step_i, "llm_raw": llm_text, "error": "invalid JSON, retrying"})
                messages.append({"role": "assistant", "content": llm_text})
                messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY a JSON action object."})
                continue

            action_type = action_data.get("action_type", "")
            parameters = action_data.get("parameters", {})

            yield _sse("thinking", {"step": step_i, "action_type": action_type, "parameters": parameters})

            # Execute in env
            try:
                action = Action(action_type=AT(action_type), parameters=parameters)
                obs = env.step(action)
            except Exception as e:
                yield _sse("step", {
                    "step": step_i, "action_type": action_type, "parameters": parameters,
                    "status": "error", "message": str(e), "reward": env.current_reward, "done": False,
                })
                messages.append({"role": "assistant", "content": json.dumps(action_data)})
                messages.append({"role": "user", "content": f"Error: {e}\n\nWhat is your next action?"})
                continue

            yield _sse("step", {
                "step": step_i,
                "action_type": action_type,
                "parameters": parameters,
                "status": obs.status,
                "message": obs.message,
                "data": obs.data,
                "reward": obs.reward,
                "done": obs.done,
                "step_number": obs.step_number,
                "max_steps": obs.max_steps,
            })

            if obs.done:
                break

            messages.append({"role": "assistant", "content": json.dumps(action_data)})
            result_text = f"Result: {obs.message}"
            if obs.data:
                result_text += f"\nData: {json.dumps(obs.data)[:600]}"
            result_text += "\n\nWhat is your next action?"
            messages.append({"role": "user", "content": result_text})

        score = grade_fn(env)
        yield _sse("done", {
            "score": round(score, 4),
            "steps_used": env.step_number,
            "max_steps": max_steps,
        })

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
