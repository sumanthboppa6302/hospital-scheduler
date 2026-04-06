"""
Inference Script — CrestView Medical Center Hospital Scheduler
===================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
    python inference.py --tasks task_easy task_medium
    python inference.py --server http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import re
import requests
from openai import OpenAI

SPACE_URL = "https://Sumanth73-hospital-scheduler.hf.space"

TASK_IDS = ["task_easy", "task_medium", "task_hard", "task_expert", "task_nightmare"]

SYSTEM_PROMPT = """You are a hospital appointment scheduling agent at CrestView Medical Center (v2.1.0).
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

Rules you MUST follow:
1. Always call get_patient_info first to understand symptoms and insurance before deciding department.
2. Always call verify_insurance before booking — some plans require referral or pre-authorization.
3. If referral required: call request_referral with a GP doctor_id before booking.
4. If pre-authorization required: call request_preauth then check_preauth_status (must be approved).
5. Always call check_availability before booking — slots may already be taken.
6. Use get_working_hours to confirm day/time before booking.
7. Book patients in triage order: emergency > urgent > waitlist > routine.
8. After cancelling, call check_waitlist_offers to catch freed slot notifications, then accept_waitlist_offer.
9. Call finish when all required appointments are completed.
10. Never repeat the same failed action — read the error and adapt.

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"action_type": "...", "parameters": {...}}"""


def get_llm_client() -> OpenAI:
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    hf_token = os.environ.get("HF_TOKEN", "")
    return OpenAI(base_url=api_base, api_key=hf_token)


def get_model() -> str:
    return os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")


def parse_action(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    # Extract first JSON object
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


def run_task(server_url: str, task_id: str, client: OpenAI, model: str) -> float:
    base = server_url.rstrip("/")

    # Reset environment
    resp = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    print(f"\n{'='*65}")
    print(f"  Task : {task_id}")
    print(f"  Model: {model}")
    print(f"{'='*65}")
    print(f"  {obs['message'].splitlines()[0][:100]}")
    print()

    max_steps = obs.get("max_steps", 30)
    messages = [
        {"role": "user", "content": f"Task:\n{obs['message']}\n\nBegin. What is your first action?"}
    ]

    for step_i in range(1, max_steps + 1):
        # Ask LLM for next action
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=256,
            temperature=0.0,
        )
        llm_text = response.choices[0].message.content or ""

        action_data = parse_action(llm_text)
        if action_data is None:
            # Ask LLM to fix its response
            messages.append({"role": "assistant", "content": llm_text})
            messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY a JSON action object."})
            continue

        action_type = action_data.get("action_type", "unknown")
        parameters = action_data.get("parameters", {})

        # Execute action
        step_resp = requests.post(
            f"{base}/step",
            json={"action": {"action_type": action_type, "parameters": parameters}},
            timeout=30,
        )
        step_resp.raise_for_status()
        obs = step_resp.json()

        icon = "[OK]" if obs["status"] == "success" else "[!!]" if obs["status"] == "warning" else "[ERR]"
        print(f"  Step {step_i:2d}: {action_type:<28} {icon} {obs['message'][:60]}")

        if obs.get("done"):
            break

        # Feed result back to LLM
        messages.append({"role": "assistant", "content": json.dumps(action_data)})
        result_text = f"Result: {obs['message']}"
        if obs.get("data"):
            result_text += f"\nData: {json.dumps(obs['data'])[:500]}"
        result_text += "\n\nWhat is your next action?"
        messages.append({"role": "user", "content": result_text})

    # Get final score
    grade_resp = requests.get(f"{base}/grade", timeout=30)
    grade_resp.raise_for_status()
    result = grade_resp.json()
    score = result.get("score", 0.0)
    print(f"\n  Score: {score:.2f} / 1.00   (steps used: {result.get('steps_used', '?')} / {max_steps})")
    return score


def main():
    parser = argparse.ArgumentParser(description="LLM inference agent for Hospital Scheduler OpenEnv")
    parser.add_argument("--server", default=SPACE_URL, help="Environment server URL")
    parser.add_argument("--tasks", nargs="*", default=TASK_IDS, help="Task IDs to run")
    args = parser.parse_args()

    # Validate env vars
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"[WARN] Missing env vars: {', '.join(missing)}")
        print("       Using defaults. Set these for proper LLM inference.\n")

    client = get_llm_client()
    model = get_model()

    print("=" * 65)
    print("  CrestView Medical Center — Hospital Scheduler")
    print("  LLM Inference Agent (OpenAI-compatible client)")
    print(f"  Model : {model}")
    print(f"  Server: {args.server}")
    print("=" * 65)

    scores: dict[str, float] = {}
    for task_id in args.tasks:
        try:
            scores[task_id] = run_task(args.server, task_id, client, model)
        except Exception as e:
            print(f"  [ERROR] {task_id}: {e}")
            scores[task_id] = 0.0

    labels = {
        "task_easy": "Easy", "task_medium": "Medium", "task_hard": "Hard",
        "task_expert": "Expert", "task_nightmare": "Nightmare",
    }
    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}")
    for tid, score in scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {labels.get(tid, tid):<12} [{bar}] {score:.2f}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average: {avg:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
