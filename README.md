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

# CrestView Medical Center — Hospital Scheduler

An OpenEnv environment where agents handle appointment scheduling at a simulated hospital. Agents must read patient records, determine the right department from symptoms, verify insurance, and book appointments — all through actions, never via direct database access.

Built for the **OpenEnv x Scalar x Hugging Face Hackathon**.

## Environment

- 7 departments: Cardiology, Orthopedics, Dermatology, Neurology, General Medicine, Emergency, Pediatrics
- 12 doctors with individual schedules, insurance acceptance, and leave dates
- 10 patients with symptoms, histories, allergies, and insurance plans
- 7 insurance providers — one is expired (agents must verify before booking)
- Waitlist system for overbooked departments
- 5 task tiers from easy to nightmare, each with partial-credit grading

## Tasks

| Task | Difficulty | Max Steps |
|------|-----------|-----------|
| Symptom-Based Booking | Easy | 10 |
| Reschedule with Constraints | Medium | 15 |
| Multi-Patient Scheduling | Hard | 25 |
| Insurance-Aware Scheduling | Expert | 20 |
| Emergency Triage | Nightmare | 35 |

## Actions (13 total)

```
search_doctors        department, date?
check_availability    doctor_id, date?
book_appointment      doctor_id, patient_id, slot_id, urgency?
cancel_appointment    appointment_id
reschedule_appointment appointment_id, new_slot_id
get_patient_info      patient_id
list_departments      (none)
get_appointment_details appointment_id
verify_insurance      patient_id, department
check_waitlist        department?, patient_id?
add_to_waitlist       patient_id, department, preferred_doctor_id?
get_doctor_schedule   doctor_id
finish                (none)
```

## Observations

```json
{
  "status": "success | warning | error",
  "message": "...",
  "data": {},
  "reward": 0.35,
  "done": false,
  "step_number": 3,
  "max_steps": 20
}
```

## API

```
GET  /          health check
POST /reset     start a task  {"task_id": "task_easy"}
POST /step      take an action
GET  /state     environment state
GET  /grade     current score
GET  /tasks     list all tasks
GET  /docs      Swagger UI
```

## Local Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

## Docker

```bash
docker build -t hospital-env .
docker run -p 7860:7860 hospital-env
```

## Project Structure

```
├── openenv.yaml
├── models.py
├── hospital_data.py
├── env.py
├── graders.py
├── inference.py
├── server/
│   ├── app.py
│   └── hospital_environment.py
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   ├── task_hard.json
│   ├── task_expert.json
│   └── task_nightmare.json
├── Dockerfile
└── requirements.txt
```
