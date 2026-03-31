"""inference.py — Run a heuristic agent against the Hospital Scheduler OpenEnv.

Usage:
    python inference.py                          # Run against live HF Space
    python inference.py --server http://localhost:7860  # Run against local server
"""

from __future__ import annotations

import argparse
import requests

SPACE_URL = "https://Sumanth73-hospital-scheduler.hf.space"

TASK_ACTIONS = {
    "task_easy": [
        {"action_type": "get_patient_info",    "parameters": {"patient_id": "P001"}},
        {"action_type": "search_doctors",      "parameters": {"department": "cardiology", "date": "2026-04-01"}},
        {"action_type": "check_availability",  "parameters": {"doctor_id": "D001", "date": "2026-04-01"}},
        {"action_type": "book_appointment",    "parameters": {"doctor_id": "D001", "patient_id": "P001", "slot_id": "D001-0401-09"}},
        {"action_type": "finish",              "parameters": {}},
    ],
    "task_medium": [
        {"action_type": "get_patient_info",       "parameters": {"patient_id": "P002"}},
        {"action_type": "get_appointment_details","parameters": {"appointment_id": "APT-101"}},
        {"action_type": "check_availability",     "parameters": {"doctor_id": "D003", "date": "2026-04-06"}},
        {"action_type": "reschedule_appointment", "parameters": {"appointment_id": "APT-101", "new_slot_id": "D003-0406-14"}},
        {"action_type": "finish",                 "parameters": {}},
    ],
    "task_hard": [
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P003"}},
        {"action_type": "search_doctors",     "parameters": {"department": "neurology", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D007", "patient_id": "P003", "slot_id": "D007-0401-09"}},
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P004"}},
        {"action_type": "search_doctors",     "parameters": {"department": "orthopedics", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D003", "patient_id": "P004", "slot_id": "D003-0401-14"}},
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P005"}},
        {"action_type": "search_doctors",     "parameters": {"department": "cardiology", "date": "2026-04-02"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D001", "patient_id": "P005", "slot_id": "D001-0402-09"}},
        {"action_type": "finish",             "parameters": {}},
    ],
    "task_expert": [
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P004"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P004", "department": "orthopedics"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P004", "department": "general_medicine"}},
        {"action_type": "search_doctors",     "parameters": {"department": "general_medicine", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D009", "patient_id": "P004", "slot_id": "D009-0401-11"}},
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P009"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P009", "department": "general_medicine"}},
        {"action_type": "search_doctors",     "parameters": {"department": "general_medicine", "date": "2026-04-02"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D009", "patient_id": "P009", "slot_id": "D009-0402-10"}},
        {"action_type": "finish",             "parameters": {}},
    ],
    "task_nightmare": [
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P008"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P008", "department": "cardiology"}},
        {"action_type": "search_doctors",     "parameters": {"department": "cardiology", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D001", "patient_id": "P008", "slot_id": "D001-0401-10", "urgency": "emergency"}},
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P007"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P007", "department": "pediatrics"}},
        {"action_type": "search_doctors",     "parameters": {"department": "pediatrics", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D012", "patient_id": "P007", "slot_id": "D012-0401-11", "urgency": "urgent"}},
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P010"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P010", "department": "neurology"}},
        {"action_type": "search_doctors",     "parameters": {"department": "neurology", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D007", "patient_id": "P010", "slot_id": "D007-0401-14"}},
        {"action_type": "get_patient_info",   "parameters": {"patient_id": "P006"}},
        {"action_type": "verify_insurance",   "parameters": {"patient_id": "P006", "department": "dermatology"}},
        {"action_type": "search_doctors",     "parameters": {"department": "dermatology", "date": "2026-04-01"}},
        {"action_type": "book_appointment",   "parameters": {"doctor_id": "D005", "patient_id": "P006", "slot_id": "D005-0401-09"}},
        {"action_type": "finish",             "parameters": {}},
    ],
}


def run_task(base_url: str, task_id: str) -> float:
    base = base_url.rstrip("/")

    resp = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")
    print(f"  Prompt: {obs['message'][:100]}...")

    for i, action in enumerate(TASK_ACTIONS[task_id], 1):
        resp = requests.post(f"{base}/step", json={"action": action}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        icon = "[OK]" if obs["status"] == "success" else "[!!]" if obs["status"] == "warning" else "[ERR]"
        print(f"  Step {i:2d}: {action['action_type']:<25} {icon} {obs['message'][:60]}")
        if obs.get("done"):
            break

    resp = requests.get(f"{base}/grade", timeout=30)
    resp.raise_for_status()
    result = resp.json()
    score = result.get("score", 0.0)
    print(f"\n  Score: {score:.2f} / 1.00   (steps used: {result.get('steps_used', '?')})")
    return score


def main():
    parser = argparse.ArgumentParser(description="Inference script for Hospital Scheduler OpenEnv")
    parser.add_argument("--server", default=SPACE_URL, help="Server URL (default: HF Space)")
    parser.add_argument("--tasks", nargs="*", default=list(TASK_ACTIONS.keys()), help="Task IDs to run")
    args = parser.parse_args()

    print("=" * 60)
    print("  CrestView Medical Center — Hospital Scheduler")
    print("  OpenEnv Heuristic Baseline Agent")
    print(f"  Server: {args.server}")
    print("=" * 60)

    scores = {}
    for task_id in args.tasks:
        try:
            scores[task_id] = run_task(args.server, task_id)
        except Exception as e:
            print(f"  [ERROR] {task_id}: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    labels = {"task_easy": "Easy", "task_medium": "Medium", "task_hard": "Hard",
              "task_expert": "Expert", "task_nightmare": "Nightmare"}
    for tid, score in scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {labels.get(tid, tid):<12} [{bar}] {score:.2f}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average: {avg:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
