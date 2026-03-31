"""Baseline inference script -- heuristic + Groq LLM agent.

Uses the Groq API client for fast LLM inference.
Reads API credentials from environment variable GROQ_API_KEY.

Usage:
    python baseline.py                    # Heuristic agent, all 5 tasks
    python baseline.py --tasks task_easy  # Single task
    python baseline.py --llm              # LLM-powered agent (needs GROQ_API_KEY)
    python baseline.py --server URL       # Remote server
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv
import requests

load_dotenv()

from env import HospitalEnv, Action, ActionType
from graders import grade


# ---------------------------------------------------------------------------
# Heuristic agents (one per task) -- now with patient lookup first
# ---------------------------------------------------------------------------

def agent_easy(env_or_url, *, remote: bool = False):
    actions = [
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P001"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "cardiology", "date": "2026-04-01"}),
        Action(action_type=ActionType.CHECK_AVAILABILITY, parameters={"doctor_id": "D001", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D001", "patient_id": "P001", "slot_id": "D001-0401-09"}),
    ]
    return _run_actions(env_or_url, "task_easy", actions, remote=remote)


def agent_medium(env_or_url, *, remote: bool = False):
    actions = [
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P002"}),
        Action(action_type=ActionType.GET_APPOINTMENT_DETAILS, parameters={"appointment_id": "APT-101"}),
        Action(action_type=ActionType.CHECK_AVAILABILITY, parameters={"doctor_id": "D003", "date": "2026-04-06"}),
        Action(action_type=ActionType.RESCHEDULE_APPOINTMENT, parameters={"appointment_id": "APT-101", "new_slot_id": "D003-0406-14"}),
    ]
    return _run_actions(env_or_url, "task_medium", actions, remote=remote)


def agent_hard(env_or_url, *, remote: bool = False):
    actions = [
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P003"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "neurology", "date": "2026-04-01"}),
        Action(action_type=ActionType.CHECK_AVAILABILITY, parameters={"doctor_id": "D007", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D007", "patient_id": "P003", "slot_id": "D007-0401-09"}),
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P004"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "orthopedics", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D003", "patient_id": "P004", "slot_id": "D003-0401-14"}),
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P005"}),
        Action(action_type=ActionType.GET_APPOINTMENT_DETAILS, parameters={"appointment_id": "APT-202"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "cardiology", "date": "2026-04-02"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D001", "patient_id": "P005", "slot_id": "D001-0402-09"}),
    ]
    return _run_actions(env_or_url, "task_hard", actions, remote=remote)


def agent_expert(env_or_url, *, remote: bool = False):
    actions = [
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P004"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P004", "department": "orthopedics"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P004", "department": "general_medicine"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "general_medicine", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D009", "patient_id": "P004", "slot_id": "D009-0401-11"}),
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P009"}),
        Action(action_type=ActionType.GET_APPOINTMENT_DETAILS, parameters={"appointment_id": "APT-404"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P009", "department": "general_medicine"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "general_medicine", "date": "2026-04-02"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D009", "patient_id": "P009", "slot_id": "D009-0402-10"}),
    ]
    return _run_actions(env_or_url, "task_expert", actions, remote=remote)


def agent_nightmare(env_or_url, *, remote: bool = False):
    actions = [
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P008"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P008", "department": "cardiology"}),
        Action(action_type=ActionType.GET_APPOINTMENT_DETAILS, parameters={"appointment_id": "APT-303"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "cardiology", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D001", "patient_id": "P008", "slot_id": "D001-0401-10", "urgency": "emergency"}),
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P007"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P007", "department": "pediatrics"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "pediatrics", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D012", "patient_id": "P007", "slot_id": "D012-0401-11", "urgency": "urgent"}),
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P010"}),
        Action(action_type=ActionType.CHECK_WAITLIST, parameters={"department": "neurology"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P010", "department": "neurology"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "neurology", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D007", "patient_id": "P010", "slot_id": "D007-0401-14"}),
        Action(action_type=ActionType.GET_PATIENT_INFO, parameters={"patient_id": "P006"}),
        Action(action_type=ActionType.VERIFY_INSURANCE, parameters={"patient_id": "P006", "department": "dermatology"}),
        Action(action_type=ActionType.SEARCH_DOCTORS, parameters={"department": "dermatology", "date": "2026-04-01"}),
        Action(action_type=ActionType.BOOK_APPOINTMENT, parameters={"doctor_id": "D005", "patient_id": "P006", "slot_id": "D005-0401-09"}),
    ]
    return _run_actions(env_or_url, "task_nightmare", actions, remote=remote)


# ---------------------------------------------------------------------------
# LLM Agent (Groq API)
# ---------------------------------------------------------------------------

def agent_llm(env_or_url, task_id: str, *, remote: bool = False):
    """Use Groq API to reason through the task."""
    try:
        from groq import Groq
    except ImportError:
        print("  [!] pip install groq  to use LLM agent")
        return 0.0

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  [!] Set GROQ_API_KEY environment variable")
        return 0.0

    client = Groq(api_key=api_key)

    if remote:
        base = env_or_url.rstrip("/")
        resp = requests.post(f"{base}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()
    else:
        obs_obj = env_or_url.reset(task_id)
        obs = {"message": obs_obj.message, "max_steps": obs_obj.max_steps, "data": obs_obj.data, "status": obs_obj.status}

    print(f"\n{'='*60}")
    print(f"  Task: {task_id} (LLM Agent - Groq)")
    print(f"{'='*60}")

    system_prompt = f"""You are a hospital appointment scheduling agent. You interact with a hospital environment by choosing actions.

Available actions: {json.dumps([a.value for a in ActionType])}

Key actions:
- get_patient_info: {{"patient_id": "P001"}} - ALWAYS start by looking up the patient's records
- search_doctors: {{"department": "...", "date": "YYYY-MM-DD"}}
- check_availability: {{"doctor_id": "...", "date": "YYYY-MM-DD"}} - ALWAYS check availability before booking
- book_appointment: {{"doctor_id": "...", "patient_id": "...", "slot_id": "..."}}
- verify_insurance: {{"patient_id": "...", "department": "..."}}
- get_appointment_details: {{"appointment_id": "..."}}
- finish: {{}} - Call this when you have completed all tasks. Do NOT keep making unnecessary actions.

IMPORTANT RULES:
1. Always look up patient records first to understand their symptoms and determine the correct department.
2. Always check_availability before booking an appointment.
3. Verify insurance before booking when dealing with insurance tasks.
4. When all required appointments are booked, call "finish" immediately. Do not waste steps.

Respond with ONLY a JSON object:
{{"action_type": "...", "parameters": {{...}}}}"""

    messages = [{"role": "user", "content": f"Task prompt:\n{obs['message']}\n\nWhat is your first action?"}]

    max_steps = obs.get("max_steps", 30)
    for step_i in range(max_steps):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        text = response.choices[0].message.content.strip()

        try:
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            action_data = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^{}]+\}', text)
            if match:
                action_data = json.loads(match.group())
            else:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "Please respond with ONLY a valid JSON action object."})
                continue

        action = Action(**action_data)

        if remote:
            resp = requests.post(f"{base}/step", json={"action": action.model_dump()})
            resp.raise_for_status()
            obs = resp.json()
        else:
            obs_obj = env_or_url.step(action)
            obs = {"message": obs_obj.message, "status": obs_obj.status, "data": obs_obj.data, "done": obs_obj.done, "reward": obs_obj.reward, "max_steps": obs_obj.max_steps, "step_number": obs_obj.step_number}

        status_icon = "[OK]" if obs["status"] == "success" else "[!!]" if obs["status"] == "warning" else "[ERR]"
        print(f"  Step {step_i+1}: {action.action_type} -> {status_icon} {obs['message'][:80]}")

        if obs.get("done"):
            break

        messages.append({"role": "assistant", "content": json.dumps(action_data)})
        messages.append({"role": "user", "content": f"Result: {obs['message']}\nData: {json.dumps(obs.get('data', {}))}\n\nWhat is your next action?"})

    if remote:
        resp = requests.get(f"{base}/grade")
        resp.raise_for_status()
        final_score = resp.json().get("score", 0.0)
    else:
        final_score = grade(env_or_url)

    print(f"\n  Final score: {final_score:.2f} / 1.00")
    return final_score


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_actions(env_or_url, task_id: str, actions: list[Action], *, remote: bool):
    if remote:
        return _run_remote(env_or_url, task_id, actions)
    return _run_local(env_or_url, task_id, actions)


def _run_local(env: HospitalEnv, task_id: str, actions: list[Action]):
    obs = env.reset(task_id)
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")
    print(f"  Prompt: {obs.message[:120]}...")
    print()

    for i, action in enumerate(actions, 1):
        obs = env.step(action)
        status_icon = "[OK]" if obs.status == "success" else "[!!]" if obs.status == "warning" else "[ERR]"
        print(f"  Step {i}: {action.action_type.value} -> {status_icon} {obs.message[:80]}")
        if obs.done:
            break

    final_score = grade(env)
    print(f"\n  Final score: {final_score:.2f} / 1.00")
    print(f"  Steps used:  {env.step_number} / {env.task_config.max_steps}")
    return final_score


def _run_remote(url: str, task_id: str, actions: list[Action]):
    base = url.rstrip("/")
    resp = requests.post(f"{base}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")
    print(f"  Prompt: {obs['message'][:120]}...")
    print()

    for i, action in enumerate(actions, 1):
        resp = requests.post(f"{base}/step", json={"action": action.model_dump()})
        resp.raise_for_status()
        obs = resp.json()
        status_icon = "[OK]" if obs["status"] == "success" else "[!!]" if obs["status"] == "warning" else "[ERR]"
        print(f"  Step {i}: {action.action_type.value} -> {status_icon} {obs['message'][:80]}")
        if obs.get("done"):
            break

    resp = requests.get(f"{base}/grade")
    resp.raise_for_status()
    result = resp.json()
    score = result["score"]
    print(f"\n  Final score: {score:.2f} / 1.00")
    print(f"  Steps used:  {result['steps_used']} / {result['max_steps']}")
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TASKS = ["task_easy", "task_medium", "task_hard", "task_expert", "task_nightmare"]

HEURISTIC_AGENTS = {
    "task_easy": agent_easy,
    "task_medium": agent_medium,
    "task_hard": agent_hard,
    "task_expert": agent_expert,
    "task_nightmare": agent_nightmare,
}


def main():
    parser = argparse.ArgumentParser(description="Baseline agent for Hospital OpenEnv")
    parser.add_argument("--server", type=str, default=None, help="Remote server URL")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task IDs to run (default: all)")
    parser.add_argument("--llm", action="store_true", help="Use Groq LLM agent instead of heuristic")
    args = parser.parse_args()

    remote = args.server is not None
    tasks = args.tasks or ALL_TASKS

    print("=" * 60)
    print("  Hospital Appointment Scheduler -- OpenEnv Baseline")
    print(f"  Agent: {'LLM (Groq)' if args.llm else 'Heuristic'}")
    print(f"  Tasks: {', '.join(tasks)}")
    print("=" * 60)

    scores = {}
    for task_id in tasks:
        target = args.server if remote else HospitalEnv()

        if args.llm:
            score = agent_llm(target, task_id, remote=remote)
        else:
            agent_fn = HEURISTIC_AGENTS.get(task_id)
            if agent_fn is None:
                print(f"\n  [!] No heuristic agent for {task_id}, skipping")
                continue
            score = agent_fn(target, remote=remote)

        scores[task_id] = score

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for tid, s in scores.items():
        filled = int(s * 20)
        bar = "#" * filled + "-" * (20 - filled)
        difficulty = {"task_easy": "Easy", "task_medium": "Medium", "task_hard": "Hard", "task_expert": "Expert", "task_nightmare": "Nightmare"}.get(tid, tid)
        print(f"  {difficulty:<12} [{bar}] {s:.2f}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average score: {avg:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
