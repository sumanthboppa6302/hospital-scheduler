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

from fastapi import HTTPException
from pydantic import BaseModel


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


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint — required by hackathon validator."""
    return {"status": "ok", "name": "hospital_scheduler", "version": "2.0.0"}


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
            ("get_patient_info", {"patient_id": "P004"}),
            ("verify_insurance", {"patient_id": "P004", "department": "orthopedics"}),
            ("verify_insurance", {"patient_id": "P004", "department": "general_medicine"}),
            ("search_doctors", {"department": "general_medicine", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D009", "patient_id": "P004", "slot_id": "D009-0401-11"}),
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
            ("verify_insurance", {"patient_id": "P010", "department": "neurology"}),
            ("search_doctors", {"department": "neurology", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D007", "patient_id": "P010", "slot_id": "D007-0401-14"}),
            ("get_patient_info", {"patient_id": "P006"}),
            ("verify_insurance", {"patient_id": "P006", "department": "dermatology"}),
            ("search_doctors", {"department": "dermatology", "date": "2026-04-01"}),
            ("book_appointment", {"doctor_id": "D005", "patient_id": "P006", "slot_id": "D005-0401-09"}),
        ],
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
