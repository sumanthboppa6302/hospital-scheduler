---
title: Hospital Scheduler
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - evaluation
  - agent
  - healthcare
---

# CrestView Medical Center — Hospital Scheduler (OpenEnv Environment)

An OpenEnv-compliant evaluation environment simulating **CrestView Medical Center**, a full-service hospital where AI agents must book, reschedule, and manage patient appointments — handling insurance validation, emergency triage, waitlist management, and multi-department coordination.

Built for the **OpenEnv x Scalar x Hugging Face Hackathon**.

---

## What Makes This Complex

This isn't a toy environment. It simulates real healthcare scheduling challenges:

- **7 departments** (Cardiology, Orthopedics, Dermatology, Neurology, General Medicine, Emergency, Pediatrics)
- **12 doctors** with specializations, ratings, insurance acceptance, leave schedules, and daily patient limits
- **10 patients** with medical histories, allergies, urgency levels, and different insurance plans
- **7 insurance providers** with varying coverage, copays, referral requirements, and one **expired plan** (trap!)
- **Waitlist system** for fully booked departments
- **Emergency triage** requiring urgency-based prioritization
- **5 tasks** from easy to nightmare difficulty, each with detailed partial-credit grading

---

## Tasks

| # | Task | Difficulty | Steps | What It Tests |
|---|------|-----------|-------|---------------|
| 1 | Simple Booking | Easy | 15 | Basic environment navigation |
| 2 | Reschedule with Constraints | Medium | 25 | Constraint reasoning (time, doctor, date range) |
| 3 | Multi-Patient Scheduling | Hard | 40 | Multi-step planning, conflict avoidance |
| 4 | Insurance-Aware Scheduling | Expert | 30 | Insurance verification, handling expired plans, referrals |
| 5 | Emergency Triage | Nightmare | 50 | Urgency triage, waitlist, insurance, pediatrics, 4 patients |

---

## Action Space (13 Actions)

| Action | Parameters | Description |
|--------|-----------|-------------|
| `search_doctors` | `department`, `date`? | Find doctors + their availability |
| `check_availability` | `doctor_id`, `date`? | Check a doctor's open slots |
| `book_appointment` | `doctor_id`, `patient_id`, `slot_id`, `urgency`? | Book an appointment |
| `cancel_appointment` | `appointment_id` | Cancel an existing appointment |
| `reschedule_appointment` | `appointment_id`, `new_slot_id` | Move appointment to a new slot |
| `get_patient_info` | `patient_id` | Patient details, history, allergies, insurance |
| `list_departments` | *(none)* | List all hospital departments |
| `get_appointment_details` | `appointment_id` | Full appointment details |
| `verify_insurance` | `patient_id`, `department` | Check insurance coverage + copay |
| `check_waitlist` | `department`?, `patient_id`? | View waitlist entries |
| `add_to_waitlist` | `patient_id`, `department`, `preferred_doctor_id`? | Add patient to waitlist |
| `get_doctor_schedule` | `doctor_id` | Full doctor schedule + leave dates |
| `finish` | *(none)* | Signal task complete — call immediately when all goals are met |

---

## Observation Space

```json
{
  "status": "success | error | warning",
  "message": "Human-readable response",
  "data": { ... },
  "available_actions": ["search_doctors", "verify_insurance", ...],
  "reward": 0.35,
  "done": false,
  "step_number": 3,
  "max_steps": 30
}
```

---

## Scoring

All tasks use partial-credit grading (0.0 -- 1.0). Each task's grader checks multiple sub-goals independently:

**Example (Nightmare task grading):**
- Emergency patient gets cardiology follow-up same/next day (+0.15)
- Emergency insurance verified (+0.05)
- Urgent child gets pediatrics within 2 days (+0.15)
- Urgent child insurance verified (+0.05)
- Waitlist checked for waitlisted patient (+0.05)
- Waitlisted patient gets neurology appointment (+0.10)
- Routine patient gets dermatology (+0.10)
- All 4 patients booked (+0.10)
- No double-bookings (+0.05)
- Correct triage order (emergency > urgent > waitlist > routine) (+0.05)

---

## Setup

### Local Development

```bash
pip install -r requirements.txt

# Run baseline (heuristic agent, all 5 tasks)
python baseline.py

# Run single task
python baseline.py --tasks task_nightmare

# Run with Groq LLM agent (needs GROQ_API_KEY in .env)
python baseline.py --llm

# Start Gradio dashboard
python app.py

# Start API server only
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t hospital-env .
docker run -p 7860:7860 hospital-env
```

### Hugging Face Spaces

Push to a HF Space with Docker SDK. The Dockerfile is pre-configured for port 7860.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Environment info |
| `POST` | `/reset` | Initialize a task |
| `POST` | `/step` | Send an action |
| `GET` | `/state` | Full environment state |
| `GET` | `/grade` | Current task score |
| `GET` | `/tasks` | List available tasks |

---

## Project Structure

```
Hospital/
├── openenv.yaml              # OpenEnv specification
├── models.py                 # Pydantic typed models
├── hospital_data.py          # Simulated hospital DB (12 doctors, 10 patients, 7 insurers)
├── env.py                    # Environment engine (13 action handlers)
├── graders.py                # 5 task-specific graders with partial credit
├── server/
│   ├── app.py                # FastAPI entry point (openenv create_app)
│   └── hospital_environment.py  # OpenEnv Environment wrapper
├── app.py                    # Gradio interactive dashboard (local demo)
├── baseline.py               # Heuristic + Groq LLM baseline agents
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   ├── task_hard.json
│   ├── task_expert.json
│   └── task_nightmare.json
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Baseline Scores (Heuristic Agent)

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| Simple Booking | Easy | 1.00 | 3/15 |
| Reschedule | Medium | 1.00 | 3/25 |
| Multi-Patient | Hard | 1.00 | 8/40 |
| Insurance-Aware | Expert | 1.00 | 8/30 |
| Emergency Triage | Nightmare | 0.90 | 14/50 |
| **Average** | | **0.97** | |

---

## Example: Interactive Agent Session

```python
import requests, json

BASE = "http://localhost:7860"

# Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "task_expert"})

# Verify insurance (discovers it's expired!)
r = requests.post(f"{BASE}/step", json={"action": {
    "action_type": "verify_insurance",
    "parameters": {"patient_id": "P004", "department": "orthopedics"}
}})
print(r.json()["message"])
# -> "Insurance ISSUE: Insurance plan 'VeriCare Standard' has expired"

# Agent discovers insurance is expired and must reroute to general_medicine...
```
