"""inference.py — Dynamic heuristic agent for the Hospital Scheduler OpenEnv.

Unlike a hardcoded script, this agent CHECKS AVAILABILITY before booking,
so it works correctly even when slots are pre-booked by other patients.

Usage:
    python inference.py                          # Run against live HF Space
    python inference.py --server http://localhost:7860  # Run against local server
    python inference.py --tasks task_easy        # Single task
"""

from __future__ import annotations

import argparse
import requests

SPACE_URL = "https://Sumanth73-hospital-scheduler.hf.space"


# ---------------------------------------------------------------------------
# Core HTTP helpers
# ---------------------------------------------------------------------------

def reset(base: str, task_id: str) -> dict:
    r = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def step(base: str, action_type: str, parameters: dict) -> dict:
    r = requests.post(f"{base}/step", json={"action": {"action_type": action_type, "parameters": parameters}}, timeout=30)
    r.raise_for_status()
    return r.json()


def grade(base: str) -> dict:
    r = requests.get(f"{base}/grade", timeout=30)
    r.raise_for_status()
    return r.json()


def print_step(i: int, action: str, obs: dict) -> None:
    icon = "[OK]" if obs["status"] == "success" else "[!!]" if obs["status"] == "warning" else "[ERR]"
    print(f"  Step {i:2d}: {action:<28} {icon} {obs['message'][:65]}")


# ---------------------------------------------------------------------------
# Dynamic helpers — discover available slots at runtime
# ---------------------------------------------------------------------------

def find_available_slot(base: str, department: str, date: str, insurance: str | None = None) -> tuple[str, str] | None:
    """Search doctors in department and return (doctor_id, slot_id) for first open slot."""
    obs = step(base, "search_doctors", {"department": department, "date": date})
    if obs["status"] != "success":
        return None
    for doc in obs["data"].get("doctors", []):
        if insurance and insurance not in doc.get("accepted_insurance", []):
            continue
        slots = doc.get("available_slots", [])
        if slots:
            return doc["doctor_id"], slots[0]["slot_id"]
    # Fallback: any doctor regardless of insurance
    for doc in obs["data"].get("doctors", []):
        slots = doc.get("available_slots", [])
        if slots:
            return doc["doctor_id"], slots[0]["slot_id"]
    return None


def find_available_slot_for_doctor(base: str, doctor_id: str, date: str) -> str | None:
    """Check availability for a specific doctor and return first open slot_id."""
    obs = step(base, "check_availability", {"doctor_id": doctor_id, "date": date})
    if obs["status"] != "success":
        return None
    slots = obs["data"].get("available_slots", [])
    return slots[0]["slot_id"] if slots else None


# ---------------------------------------------------------------------------
# Task agents
# ---------------------------------------------------------------------------

def agent_easy(base: str) -> None:
    """Book earliest cardiology slot for P001. Must check availability — 09:00 is pre-booked."""
    i = 0
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P001"}); print_step(i, "get_patient_info P001", obs)
    # Dynamically find first open cardiology slot on 2026-04-01
    result = find_available_slot(base, "cardiology", "2026-04-01", insurance="AzureShield")
    i += 1; print_step(i, "search_doctors cardiology", {"status": "success", "message": f"Found slot: {result}"})
    if result:
        doctor_id, slot_id = result
        i += 1; obs = step(base, "check_availability", {"doctor_id": doctor_id, "date": "2026-04-01"}); print_step(i, f"check_availability {doctor_id}", obs)
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doctor_id, "patient_id": "P001", "slot_id": slot_id}); print_step(i, "book_appointment P001", obs)
    i += 1; obs = step(base, "finish", {}); print_step(i, "finish", obs)


def agent_medium(base: str) -> None:
    """Reschedule APT-101 to afternoon slot next week with same doctor."""
    i = 0
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P002"}); print_step(i, "get_patient_info P002", obs)
    i += 1; obs = step(base, "get_appointment_details", {"appointment_id": "APT-101"}); print_step(i, "get_appointment_details APT-101", obs)
    # Find afternoon slot for D003 in date range 2026-04-06 to 2026-04-10
    target_slot = None
    for date in ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"]:
        slot_id = find_available_slot_for_doctor(base, "D003", date)
        if slot_id and slot_id.split("-")[-1] >= "14":  # afternoon (14:xx or later)
            target_slot = slot_id
            break
        elif slot_id is None:
            # check_availability already called inside find_available_slot_for_doctor
            pass
    i += 1; obs = step(base, "check_availability", {"doctor_id": "D003", "date": "2026-04-06"}); print_step(i, "check_availability D003", obs)
    # Pick the first afternoon slot found
    if not target_slot:
        for slot in obs["data"].get("available_slots", []):
            if slot["start_time"] >= "14:00":
                target_slot = slot["slot_id"]
                break
    if target_slot:
        i += 1; obs = step(base, "reschedule_appointment", {"appointment_id": "APT-101", "new_slot_id": target_slot}); print_step(i, "reschedule_appointment APT-101", obs)
    i += 1; obs = step(base, "finish", {}); print_step(i, "finish", obs)


def agent_hard(base: str) -> None:
    """Book 3 patients: P003 (neurology, urgent), P004 (orthopedics), P005 (cardiology after APT-202)."""
    i = 0
    # P003 — urgent neurology (D007-0401-09 is pre-booked, must find next available)
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P003"}); print_step(i, "get_patient_info P003", obs)
    result = find_available_slot(base, "neurology", "2026-04-01", insurance="UnionCare")
    i += 1; print_step(i, "search_doctors neurology", {"status": "success", "message": f"Found: {result}"})
    if result:
        doc_id, slot_id = result
        i += 1; obs = step(base, "check_availability", {"doctor_id": doc_id, "date": "2026-04-01"}); print_step(i, f"check_availability {doc_id}", obs)
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id, "patient_id": "P003", "slot_id": slot_id}); print_step(i, "book_appointment P003", obs)

    # P004 — orthopedics, prefers D003
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P004"}); print_step(i, "get_patient_info P004", obs)
    slot_id = find_available_slot_for_doctor(base, "D003", "2026-04-01")
    i += 1; print_step(i, "check_availability D003", {"status": "success", "message": f"slot={slot_id}"})
    if not slot_id:
        result2 = find_available_slot(base, "orthopedics", "2026-04-01")
        slot_id = result2[1] if result2 else None
        doc_id2 = result2[0] if result2 else "D003"
    else:
        doc_id2 = "D003"
    if slot_id:
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id2, "patient_id": "P004", "slot_id": slot_id}); print_step(i, "book_appointment P004", obs)

    # P005 — cardiology after APT-202 (2026-04-01), so use 2026-04-02
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P005"}); print_step(i, "get_patient_info P005", obs)
    result3 = find_available_slot(base, "cardiology", "2026-04-02", insurance="FederalMed")
    i += 1; print_step(i, "search_doctors cardiology 04-02", {"status": "success", "message": f"Found: {result3}"})
    if result3:
        doc_id3, slot_id3 = result3
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id3, "patient_id": "P005", "slot_id": slot_id3}); print_step(i, "book_appointment P005", obs)
    i += 1; obs = step(base, "finish", {}); print_step(i, "finish", obs)


def agent_expert(base: str) -> None:
    """Insurance-aware booking: P004 (VeriCare EXPIRED → general_medicine), P009 (GuardianPlan → general_medicine)."""
    i = 0
    # P004 — VeriCare is EXPIRED, must fallback to general_medicine
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P004"}); print_step(i, "get_patient_info P004", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P004", "department": "orthopedics"}); print_step(i, "verify_insurance P004 ortho", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P004", "department": "general_medicine"}); print_step(i, "verify_insurance P004 gen_med", obs)
    result = find_available_slot(base, "general_medicine", "2026-04-01", insurance="VeriCare")
    i += 1; print_step(i, "search_doctors general_medicine", {"status": "success", "message": f"Found: {result}"})
    if result:
        doc_id, slot_id = result
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id, "patient_id": "P004", "slot_id": slot_id}); print_step(i, "book_appointment P004", obs)

    # P009 — check referral then book with GuardianPlan-accepting doctor
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P009"}); print_step(i, "get_patient_info P009", obs)
    i += 1; obs = step(base, "get_appointment_details", {"appointment_id": "APT-404"}); print_step(i, "get_appointment_details APT-404", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P009", "department": "general_medicine"}); print_step(i, "verify_insurance P009 gen_med", obs)
    result2 = find_available_slot(base, "general_medicine", "2026-04-02", insurance="GuardianPlan")
    i += 1; print_step(i, "search_doctors general_medicine 04-02", {"status": "success", "message": f"Found: {result2}"})
    if result2:
        doc_id2, slot_id2 = result2
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id2, "patient_id": "P009", "slot_id": slot_id2}); print_step(i, "book_appointment P009", obs)
    i += 1; obs = step(base, "finish", {}); print_step(i, "finish", obs)


def agent_nightmare(base: str) -> None:
    """Emergency triage: P008 (emergency) → P007 (urgent) → P010 (waitlist) → P006 (routine)."""
    i = 0
    # 1. P008 — EMERGENCY cardiology (APT-303 exists, need follow-up)
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P008"}); print_step(i, "get_patient_info P008", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P008", "department": "cardiology"}); print_step(i, "verify_insurance P008 cardio", obs)
    result = find_available_slot(base, "cardiology", "2026-04-01", insurance="FederalMed")
    i += 1; print_step(i, "search_doctors cardiology", {"status": "success", "message": f"Found: {result}"})
    if result:
        doc_id, slot_id = result
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id, "patient_id": "P008", "slot_id": slot_id, "urgency": "emergency"}); print_step(i, "book_appointment P008 EMERGENCY", obs)

    # 2. P007 — URGENT pediatrics (D012-0401-09 pre-booked, must find next slot)
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P007"}); print_step(i, "get_patient_info P007", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P007", "department": "pediatrics"}); print_step(i, "verify_insurance P007 peds", obs)
    result2 = find_available_slot(base, "pediatrics", "2026-04-01", insurance="StateWell")
    i += 1; print_step(i, "search_doctors pediatrics", {"status": "success", "message": f"Found: {result2}"})
    if result2:
        doc_id2, slot_id2 = result2
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id2, "patient_id": "P007", "slot_id": slot_id2, "urgency": "urgent"}); print_step(i, "book_appointment P007 URGENT", obs)

    # 3. P010 — check waitlist first, then book neurology
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P010"}); print_step(i, "get_patient_info P010", obs)
    i += 1; obs = step(base, "check_waitlist", {"department": "neurology"}); print_step(i, "check_waitlist neurology", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P010", "department": "neurology"}); print_step(i, "verify_insurance P010 neuro", obs)
    result3 = find_available_slot(base, "neurology", "2026-04-01", insurance="UnionCare")
    i += 1; print_step(i, "search_doctors neurology", {"status": "success", "message": f"Found: {result3}"})
    if result3:
        doc_id3, slot_id3 = result3
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id3, "patient_id": "P010", "slot_id": slot_id3}); print_step(i, "book_appointment P010", obs)

    # 4. P006 — ROUTINE dermatology
    i += 1; obs = step(base, "get_patient_info", {"patient_id": "P006"}); print_step(i, "get_patient_info P006", obs)
    i += 1; obs = step(base, "verify_insurance", {"patient_id": "P006", "department": "dermatology"}); print_step(i, "verify_insurance P006 derm", obs)
    result4 = find_available_slot(base, "dermatology", "2026-04-01", insurance="AzureShield")
    i += 1; print_step(i, "search_doctors dermatology", {"status": "success", "message": f"Found: {result4}"})
    if result4:
        doc_id4, slot_id4 = result4
        i += 1; obs = step(base, "book_appointment", {"doctor_id": doc_id4, "patient_id": "P006", "slot_id": slot_id4}); print_step(i, "book_appointment P006", obs)

    i += 1; obs = step(base, "finish", {}); print_step(i, "finish", obs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

AGENTS = {
    "task_easy":      agent_easy,
    "task_medium":    agent_medium,
    "task_hard":      agent_hard,
    "task_expert":    agent_expert,
    "task_nightmare": agent_nightmare,
}

LABELS = {
    "task_easy": "Easy", "task_medium": "Medium", "task_hard": "Hard",
    "task_expert": "Expert", "task_nightmare": "Nightmare",
}


def run_task(base_url: str, task_id: str) -> float:
    base = base_url.rstrip("/")
    obs = reset(base, task_id)
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}  ({LABELS.get(task_id, task_id)})")
    print(f"{'='*60}")
    print(f"  {obs['message'][:120].splitlines()[0]}")

    AGENTS[task_id](base)

    result = grade(base)
    score = result.get("score", 0.0)
    print(f"\n  Score: {score:.2f} / 1.00   steps used: {result.get('steps_used', '?')}")
    return score


def main():
    parser = argparse.ArgumentParser(description="Inference script for Hospital Scheduler OpenEnv")
    parser.add_argument("--server", default=SPACE_URL, help="Server URL (default: HF Space)")
    parser.add_argument("--tasks", nargs="*", default=list(AGENTS.keys()), help="Task IDs to run")
    args = parser.parse_args()

    print("=" * 60)
    print("  CrestView Medical Center — Hospital Scheduler")
    print("  Dynamic Heuristic Baseline Agent")
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
    for tid, score in scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {LABELS.get(tid, tid):<12} [{bar}] {score:.2f}")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average: {avg:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
