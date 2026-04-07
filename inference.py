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

STDOUT FORMAT (required by hackathon validator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import requests
from openai import OpenAI

SPACE_URL = "https://Sumanth73-hospital-scheduler.hf.space"
ENV_BENCHMARK = "hospital_scheduler"
SUCCESS_THRESHOLD = 0.1

# ── Required env-var variables (checklist: only API_BASE_URL and MODEL_NAME have defaults) ──
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")          # intentionally no default

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


# ── Structured output helpers ──────────────────────────────────────────────

def _log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_BENCHMARK} model={model}", flush=True)


def _log_step(step: int, action_type: str, parameters: dict,
              reward: float, done: bool, error: str | None) -> None:
    action_str = f"{action_type}({json.dumps(parameters, separators=(',', ':'))})"
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ────────────────────────────────────────────────────────────

def get_llm_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "")


def get_model() -> str:
    return MODEL_NAME


def parse_action(text: str) -> dict | None:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("action_type"):
            return data
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object using balanced brace matching
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start:i + 1])
                    if isinstance(data, dict) and data.get("action_type"):
                        return data
                except json.JSONDecodeError:
                    pass
                break
    return None


def _unwrap_obs(raw: dict) -> dict:
    """Handle both flat and OpenEnv-wrapped observation formats."""
    if "observation" in raw and isinstance(raw["observation"], dict):
        obs = raw["observation"]
        obs.setdefault("reward", raw.get("reward", 0.0))
        obs.setdefault("done", raw.get("done", False))
        return obs
    return raw


# ── Task runner ────────────────────────────────────────────────────────────

def run_task(server_url: str, task_id: str, client: OpenAI, model: str) -> float:
    base = server_url.rstrip("/")

    rewards: list[float] = []
    steps_used = 0
    score = 0.0
    success = False

    _log_start(task_id, model)

    try:
        resp = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = _unwrap_obs(resp.json())

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
                messages.append({"role": "assistant", "content": llm_text})
                messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY a JSON action object."})
                continue

            action_type = action_data.get("action_type", "unknown")
            parameters = action_data.get("parameters", {})

            # Execute action in environment
            step_error: str | None = None
            try:
                step_resp = requests.post(
                    f"{base}/step",
                    json={"action": {"action_type": action_type, "parameters": parameters}},
                    timeout=30,
                )
                step_resp.raise_for_status()
                obs = _unwrap_obs(step_resp.json())
            except Exception as exc:
                step_error = str(exc)
                _log_step(step_i, action_type, parameters, 0.0, False, step_error)
                rewards.append(0.0)
                steps_used = step_i
                break

            step_reward = max(0.01, min(0.99, float(obs.get("reward", 0.0))))
            done = bool(obs.get("done", False))

            # Treat env-level errors as non-null error field
            if obs.get("status") == "error":
                step_error = obs.get("message", "error")[:120]

            rewards.append(step_reward)
            steps_used = step_i

            icon = "[OK]" if obs.get("status") == "success" else \
                   "[!!]" if obs.get("status") == "warning" else "[ERR]"
            print(f"  Step {step_i:2d}: {action_type:<28} {icon} {obs.get('message', '')[:60]}")

            _log_step(step_i, action_type, parameters, step_reward, done, step_error)

            if done:
                break

            messages.append({"role": "assistant", "content": json.dumps(action_data)})
            result_text = f"Result: {obs['message']}"
            if obs.get("data"):
                result_text += f"\nData: {json.dumps(obs['data'])[:500]}"
            result_text += "\n\nWhat is your next action?"
            messages.append({"role": "user", "content": result_text})

        # Score: try /grade endpoint, fall back to last observed reward
        score = rewards[-1] if rewards else 0.0
        try:
            grade_resp = requests.get(f"{base}/grade", timeout=15)
            if grade_resp.ok:
                result = grade_resp.json()
                if isinstance(result.get("score"), (int, float)):
                    score = float(result["score"])
                    steps_used = result.get("steps_used", steps_used)
        except Exception:
            pass

        # Clamp strictly inside (0, 1) — validator rejects exactly 0.0 or 1.0
        score = max(0.01, min(0.99, score))
        print(f"\n  Score: {score:.4f} / 1.00   (steps used: {steps_used} / {max_steps})")
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"  [ERROR] {task_id}: {exc}")
        score = 0.01   # minimum non-zero score (never 0.0)
        success = False

    finally:
        _log_end(success, steps_used, score, rewards)

    return score


def main():
    parser = argparse.ArgumentParser(description="LLM inference agent for Hospital Scheduler OpenEnv")
    parser.add_argument("--server", default=SPACE_URL, help="Environment server URL")
    parser.add_argument("--tasks", nargs="*", default=TASK_IDS, help="Task IDs to run")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set — LLM calls may fail without a valid API key.\n")

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
