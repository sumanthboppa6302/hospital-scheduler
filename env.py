"""Hospital Appointment Scheduling -- OpenEnv environment implementation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from models import (
    ActionType,
    Appointment,
    AppointmentStatus,
    Department,
    TaskConfig,
    TimeSlot,
    UrgencyLevel,
    WaitlistEntry,
)

# Internal Action/Observation models (used by env logic and baseline)
from pydantic import BaseModel


class Action(BaseModel):
    action_type: ActionType
    parameters: dict[str, Any] = {}


class Observation(BaseModel):
    status: str = "success"
    message: str = ""
    data: dict[str, Any] = {}
    available_actions: list[str] = []
    reward: float = 0.0
    done: bool = False
    step_number: int = 0
    max_steps: int = 0


class EnvState(BaseModel):
    task_id: str = ""
    task_prompt: str = ""
    step_number: int = 0
    max_steps: int = 0
    done: bool = False
    current_reward: float = 0.0
    appointments: list[Appointment] = []
    waitlist: list[WaitlistEntry] = []
    action_history: list[Action] = []
    observation_history: list[Observation] = []


from hospital_data import (
    DOCTORS,
    PATIENTS,
    SEED_APPOINTMENTS,
    SEED_WAITLIST,
    check_insurance_coverage,
    get_doctor,
    get_doctors_by_department,
    get_insurance_plan,
    get_patient,
)

TASKS_DIR = Path(__file__).parent / "tasks"
ALL_ACTION_NAMES = [a.value for a in ActionType]


class HospitalEnv:
    """OpenEnv-compliant hospital appointment scheduling environment."""

    def __init__(self) -> None:
        self.task_config: TaskConfig | None = None
        self.appointments: list[Appointment] = []
        self.waitlist: list[WaitlistEntry] = []
        self.booked_slots: set[str] = set()
        self.action_history: list[Action] = []
        self.observation_history: list[Observation] = []
        self.step_number: int = 0
        self.done: bool = False
        self.current_reward: float = 0.0
        self.doctors = copy.deepcopy(DOCTORS)
        self.insurance_verified: dict[str, dict] = {}
        self._consecutive_errors: int = 0
        self._repeated_actions: int = 0
        self._last_action_key: str = ""
        self._patients_looked_up: set[str] = set()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        task_path = TASKS_DIR / f"{task_id}.json"
        if not task_path.exists():
            return Observation(status="error", message=f"Unknown task: {task_id}", available_actions=ALL_ACTION_NAMES)

        with open(task_path, "r") as f:
            raw = json.load(f)
        self.task_config = TaskConfig(**raw)

        self.doctors = copy.deepcopy(DOCTORS)
        self.appointments = []
        self.waitlist = []
        self.booked_slots = set()
        self.action_history = []
        self.observation_history = []
        self.step_number = 0
        self.done = False
        self.current_reward = 0.0
        self.insurance_verified = {}
        self._consecutive_errors = 0
        self._repeated_actions = 0
        self._last_action_key = ""
        self._patients_looked_up = set()

        for appt_id in self.task_config.initial_appointments:
            for seed in SEED_APPOINTMENTS:
                if seed.appointment_id == appt_id:
                    appt = seed.model_copy(deep=True)
                    self.appointments.append(appt)
                    self.booked_slots.add(appt.slot.slot_id)
                    self._mark_slot_unavailable(appt.slot.slot_id)

        if task_id == "task_nightmare":
            for entry in SEED_WAITLIST:
                self.waitlist.append(entry.model_copy(deep=True))

        if task_id == "task_medium":
            self._add_next_week_slots()

        obs = Observation(
            status="success",
            message=f"Environment reset for task: {self.task_config.task_id}\n\n{self.task_config.prompt}",
            data={"task_id": self.task_config.task_id, "difficulty": self.task_config.difficulty, "available_actions": ALL_ACTION_NAMES},
            available_actions=ALL_ACTION_NAMES,
            reward=0.0, done=False, step_number=0, max_steps=self.task_config.max_steps,
        )
        self.observation_history.append(obs)
        return obs

    def step(self, action: Action) -> Observation:
        if self.task_config is None:
            return Observation(status="error", message="Environment not initialized. Call reset() first.")
        if self.done:
            return Observation(status="error", message="Episode is already done.", reward=self.current_reward, done=True, step_number=self.step_number, max_steps=self.task_config.max_steps)

        self.step_number += 1
        self.action_history.append(action)

        # Track repeated actions for penalty
        action_key = f"{action.action_type.value}:{json.dumps(action.parameters, sort_keys=True)}"
        if action_key == self._last_action_key:
            self._repeated_actions += 1
        else:
            self._repeated_actions = 0
        self._last_action_key = action_key

        handler = {
            ActionType.SEARCH_DOCTORS: self._handle_search_doctors,
            ActionType.CHECK_AVAILABILITY: self._handle_check_availability,
            ActionType.BOOK_APPOINTMENT: self._handle_book_appointment,
            ActionType.CANCEL_APPOINTMENT: self._handle_cancel_appointment,
            ActionType.RESCHEDULE_APPOINTMENT: self._handle_reschedule_appointment,
            ActionType.GET_PATIENT_INFO: self._handle_get_patient_info,
            ActionType.LIST_DEPARTMENTS: self._handle_list_departments,
            ActionType.GET_APPOINTMENT_DETAILS: self._handle_get_appointment_details,
            ActionType.VERIFY_INSURANCE: self._handle_verify_insurance,
            ActionType.CHECK_WAITLIST: self._handle_check_waitlist,
            ActionType.ADD_TO_WAITLIST: self._handle_add_to_waitlist,
            ActionType.GET_DOCTOR_SCHEDULE: self._handle_get_doctor_schedule,
            ActionType.FINISH: self._handle_finish,
        }.get(action.action_type)

        if handler is None:
            obs = Observation(status="error", message=f"Unknown action: {action.action_type}", available_actions=ALL_ACTION_NAMES, step_number=self.step_number, max_steps=self.task_config.max_steps)
        else:
            obs = handler(action.parameters)

        # Penalty: repeated identical actions (loop detection)
        if self._repeated_actions >= 3:
            penalty = min(0.05 * (self._repeated_actions - 2), 0.15)
            self.current_reward = max(0.0, self.current_reward - penalty)
            obs.message += f" [PENALTY: -{penalty:.2f} for repeating the same action {self._repeated_actions + 1} times]"

        # Penalty: consecutive errors
        if obs.status == "error":
            self._consecutive_errors += 1
            if self._consecutive_errors >= 5:
                penalty = 0.05
                self.current_reward = max(0.0, self.current_reward - penalty)
                obs.message += f" [PENALTY: -{penalty:.2f} for {self._consecutive_errors} consecutive errors]"
        else:
            self._consecutive_errors = 0

        if self.step_number >= self.task_config.max_steps:
            self.done = True
            obs.done = True
            obs.message += "\n[Max steps reached -- episode ended.]"

        obs.step_number = self.step_number
        obs.max_steps = self.task_config.max_steps
        obs.reward = self.current_reward
        self.observation_history.append(obs)
        return obs

    def state(self) -> EnvState:
        return EnvState(
            task_id=self.task_config.task_id if self.task_config else "",
            task_prompt=self.task_config.prompt if self.task_config else "",
            step_number=self.step_number,
            max_steps=self.task_config.max_steps if self.task_config else 0,
            done=self.done, current_reward=self.current_reward,
            appointments=self.appointments, waitlist=self.waitlist,
            action_history=self.action_history, observation_history=self.observation_history,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_search_doctors(self, params: dict[str, Any]) -> Observation:
        department = params.get("department", "")
        date = params.get("date", "")
        doctors = get_doctors_by_department(department)
        if not doctors:
            return Observation(status="error", message=f"No doctors found in '{department}'. Valid: {[d.value for d in Department]}", available_actions=ALL_ACTION_NAMES)
        results = []
        for doc_template in doctors:
            doc = self._get_doctor(doc_template.doctor_id)
            if doc is None:
                continue
            available_slots = [s for s in doc.availability if s.is_available and (not date or s.date == date) and s.date not in doc.on_leave_dates]
            results.append({
                "doctor_id": doc.doctor_id, "name": doc.name, "specialization": doc.specialization, "rating": doc.rating,
                "accepted_insurance": doc.accepted_insurance, "available_slots_count": len(available_slots),
                "available_slots": [{"slot_id": s.slot_id, "date": s.date, "start_time": s.start_time, "end_time": s.end_time} for s in available_slots],
                "on_leave_dates": doc.on_leave_dates,
            })
        self._update_reward_for_search(department)
        return Observation(status="success", message=f"Found {len(results)} doctor(s) in {department}" + (f" on {date}" if date else ""), data={"doctors": results}, available_actions=ALL_ACTION_NAMES)

    def _handle_check_availability(self, params: dict[str, Any]) -> Observation:
        doctor_id = params.get("doctor_id", "")
        date = params.get("date", "")
        doc = self._get_doctor(doctor_id)
        if doc is None:
            return Observation(status="error", message=f"Doctor '{doctor_id}' not found.", available_actions=ALL_ACTION_NAMES)
        available = [{"slot_id": s.slot_id, "date": s.date, "start_time": s.start_time, "end_time": s.end_time}
                     for s in doc.availability if s.is_available and (not date or s.date == date) and s.date not in doc.on_leave_dates]
        self._update_reward_for_availability_check()
        return Observation(status="success", message=f"{doc.name} has {len(available)} available slot(s)" + (f" on {date}" if date else ""),
                           data={"doctor_id": doctor_id, "doctor_name": doc.name, "department": doc.department.value, "accepted_insurance": doc.accepted_insurance, "available_slots": available, "on_leave_dates": doc.on_leave_dates}, available_actions=ALL_ACTION_NAMES)

    def _handle_book_appointment(self, params: dict[str, Any]) -> Observation:
        doctor_id = params.get("doctor_id", "")
        patient_id = params.get("patient_id", "")
        slot_id = params.get("slot_id", "")
        urgency = params.get("urgency", "routine")
        doc = self._get_doctor(doctor_id)
        if doc is None:
            return Observation(status="error", message=f"Doctor '{doctor_id}' not found.", available_actions=ALL_ACTION_NAMES)
        patient = get_patient(patient_id)
        if patient is None:
            return Observation(status="error", message=f"Patient '{patient_id}' not found.", available_actions=ALL_ACTION_NAMES)
        slot = next((s for s in doc.availability if s.slot_id == slot_id), None)
        if slot is None:
            return Observation(status="error", message=f"Slot '{slot_id}' not found for {doctor_id}.", available_actions=ALL_ACTION_NAMES)
        if not slot.is_available or slot_id in self.booked_slots:
            return Observation(status="error", message=f"Slot '{slot_id}' is not available.", available_actions=ALL_ACTION_NAMES)
        if slot.date in doc.on_leave_dates:
            return Observation(status="error", message=f"{doc.name} is on leave on {slot.date}.", available_actions=ALL_ACTION_NAMES)

        insurance_warning = ""
        if self.task_config and self.task_config.constraints.get("insurance_check_required"):
            if patient_id not in self.insurance_verified:
                insurance_warning = " WARNING: Insurance not verified before booking."

        appt_id = f"APT-{len(self.appointments) + 500}"
        try:
            urg = UrgencyLevel(urgency)
        except ValueError:
            urg = UrgencyLevel.ROUTINE

        appt = Appointment(
            appointment_id=appt_id, patient_id=patient_id, doctor_id=doctor_id,
            slot=slot.model_copy(), status=AppointmentStatus.SCHEDULED,
            urgency=urg, insurance_verified=patient_id in self.insurance_verified,
        )
        appt.slot.is_available = False
        self.appointments.append(appt)
        self.booked_slots.add(slot_id)
        self._mark_slot_unavailable(slot_id)
        self.waitlist = [w for w in self.waitlist if w.patient_id != patient_id]
        self._update_reward_for_booking(patient_id, doctor_id, slot)

        return Observation(
            status="success",
            message=f"Appointment {appt_id} booked: {patient.name} with {doc.name} on {slot.date} at {slot.start_time}.{insurance_warning}",
            data={"appointment_id": appt_id, "patient_id": patient_id, "doctor_id": doctor_id, "doctor_name": doc.name,
                  "slot": {"slot_id": slot_id, "date": slot.date, "start_time": slot.start_time, "end_time": slot.end_time},
                  "urgency": urg.value, "insurance_verified": patient_id in self.insurance_verified},
            available_actions=ALL_ACTION_NAMES,
        )

    def _handle_cancel_appointment(self, params: dict[str, Any]) -> Observation:
        appt_id = params.get("appointment_id", "")
        appt = next((a for a in self.appointments if a.appointment_id == appt_id), None)
        if appt is None:
            return Observation(status="error", message=f"Appointment '{appt_id}' not found.", available_actions=ALL_ACTION_NAMES)
        appt.status = AppointmentStatus.CANCELLED
        self.booked_slots.discard(appt.slot.slot_id)
        self._mark_slot_available(appt.slot.slot_id)
        return Observation(status="success", message=f"Appointment {appt_id} cancelled.", data={"appointment_id": appt_id, "status": "cancelled"}, available_actions=ALL_ACTION_NAMES)

    def _handle_reschedule_appointment(self, params: dict[str, Any]) -> Observation:
        appt_id = params.get("appointment_id", "")
        new_slot_id = params.get("new_slot_id", "")
        appt = next((a for a in self.appointments if a.appointment_id == appt_id), None)
        if appt is None:
            return Observation(status="error", message=f"Appointment '{appt_id}' not found.", available_actions=ALL_ACTION_NAMES)
        doc = self._get_doctor(appt.doctor_id)
        if doc is None:
            return Observation(status="error", message="Doctor no longer available.", available_actions=ALL_ACTION_NAMES)
        new_slot = next((s for s in doc.availability if s.slot_id == new_slot_id), None)
        if new_slot is None:
            return Observation(status="error", message=f"Slot '{new_slot_id}' not found for {doc.name}.", available_actions=ALL_ACTION_NAMES)
        if not new_slot.is_available or new_slot_id in self.booked_slots:
            return Observation(status="error", message=f"Slot '{new_slot_id}' is not available.", available_actions=ALL_ACTION_NAMES)
        if new_slot.date in doc.on_leave_dates:
            return Observation(status="error", message=f"{doc.name} is on leave on {new_slot.date}.", available_actions=ALL_ACTION_NAMES)

        self.booked_slots.discard(appt.slot.slot_id)
        self._mark_slot_available(appt.slot.slot_id)
        appt.slot = new_slot.model_copy()
        appt.slot.is_available = False
        appt.status = AppointmentStatus.RESCHEDULED
        self.booked_slots.add(new_slot_id)
        self._mark_slot_unavailable(new_slot_id)
        self._update_reward_for_reschedule(appt, new_slot)
        return Observation(status="success", message=f"Appointment {appt_id} rescheduled to {new_slot.date} at {new_slot.start_time} with {doc.name}.",
                           data={"appointment_id": appt_id, "new_slot": {"slot_id": new_slot_id, "date": new_slot.date, "start_time": new_slot.start_time, "end_time": new_slot.end_time}, "doctor_name": doc.name}, available_actions=ALL_ACTION_NAMES)

    def _handle_get_patient_info(self, params: dict[str, Any]) -> Observation:
        patient_id = params.get("patient_id", "")
        patient = get_patient(patient_id)
        if patient is None:
            return Observation(status="error", message=f"Patient '{patient_id}' not found.", available_actions=ALL_ACTION_NAMES)

        self._patients_looked_up.add(patient_id)
        self._update_reward_for_patient_lookup(patient_id)

        existing_appts = [
            {"appointment_id": a.appointment_id, "doctor_id": a.doctor_id, "date": a.slot.date, "start_time": a.slot.start_time, "status": a.status.value, "urgency": a.urgency.value}
            for a in self.appointments if a.patient_id == patient_id and a.status in (AppointmentStatus.SCHEDULED, AppointmentStatus.RESCHEDULED)
        ]
        waitlist_entries = [{"department": w.department, "preferred_doctor": w.preferred_doctor_id, "date_added": w.date_added} for w in self.waitlist if w.patient_id == patient_id]

        return Observation(status="success", message=f"Patient info for {patient.name}.",
                           data={"patient_id": patient.patient_id, "name": patient.name, "age": patient.age, "symptoms": patient.symptoms,
                                 "insurance": patient.insurance, "urgency": patient.urgency.value, "allergies": patient.allergies,
                                 "medical_history": patient.medical_history, "existing_appointments": existing_appts, "waitlist_entries": waitlist_entries},
                           available_actions=ALL_ACTION_NAMES)

    def _handle_list_departments(self, _params: dict[str, Any]) -> Observation:
        departments = [{"name": d.value, "label": d.value.replace("_", " ").title()} for d in Department]
        return Observation(status="success", message="Available hospital departments.", data={"departments": departments}, available_actions=ALL_ACTION_NAMES)

    def _handle_get_appointment_details(self, params: dict[str, Any]) -> Observation:
        appt_id = params.get("appointment_id", "")
        appt = next((a for a in self.appointments if a.appointment_id == appt_id), None)
        if appt is None:
            return Observation(status="error", message=f"Appointment '{appt_id}' not found.", available_actions=ALL_ACTION_NAMES)
        doc = self._get_doctor(appt.doctor_id)
        patient = get_patient(appt.patient_id)
        self._update_reward_for_detail_retrieval()
        return Observation(status="success", message=f"Details for appointment {appt_id}.",
                           data={"appointment_id": appt.appointment_id, "patient_id": appt.patient_id, "patient_name": patient.name if patient else "Unknown",
                                 "doctor_id": appt.doctor_id, "doctor_name": doc.name if doc else "Unknown", "department": doc.department.value if doc else "Unknown",
                                 "date": appt.slot.date, "start_time": appt.slot.start_time, "end_time": appt.slot.end_time,
                                 "status": appt.status.value, "urgency": appt.urgency.value, "notes": appt.notes, "insurance_verified": appt.insurance_verified},
                           available_actions=ALL_ACTION_NAMES)

    def _handle_verify_insurance(self, params: dict[str, Any]) -> Observation:
        patient_id = params.get("patient_id", "")
        department = params.get("department", "")
        patient = get_patient(patient_id)
        if patient is None:
            return Observation(status="error", message=f"Patient '{patient_id}' not found.", available_actions=ALL_ACTION_NAMES)
        coverage = check_insurance_coverage(patient_id, department)
        self.insurance_verified[patient_id] = coverage
        self._update_reward_for_insurance_check(patient_id, coverage)
        if coverage["covered"]:
            msg = f"Insurance VERIFIED: {patient.name}'s {coverage['provider']} covers {department}. Copay: ${coverage['copay']:.0f}."
            if coverage.get("requires_referral"):
                msg += " NOTE: Referral required for specialist visits."
            return Observation(status="success", message=msg, data=coverage, available_actions=ALL_ACTION_NAMES)
        else:
            return Observation(status="warning", message=f"Insurance ISSUE: {coverage['reason']}", data=coverage, available_actions=ALL_ACTION_NAMES)

    def _handle_check_waitlist(self, params: dict[str, Any]) -> Observation:
        department = params.get("department", "")
        patient_id = params.get("patient_id", "")
        entries = self.waitlist
        if department:
            entries = [w for w in entries if w.department == department.lower().replace(" ", "_")]
        if patient_id:
            entries = [w for w in entries if w.patient_id == patient_id]
        result = [{"patient_id": w.patient_id, "department": w.department, "preferred_doctor_id": w.preferred_doctor_id,
                    "date_added": w.date_added, "urgency": w.urgency.value, "notes": w.notes} for w in entries]
        self._update_reward_for_waitlist_check()
        return Observation(status="success", message=f"Found {len(result)} waitlist entry(ies)" + (f" for {department}" if department else ""),
                           data={"waitlist": result}, available_actions=ALL_ACTION_NAMES)

    def _handle_add_to_waitlist(self, params: dict[str, Any]) -> Observation:
        patient_id = params.get("patient_id", "")
        department = params.get("department", "")
        preferred_doctor_id = params.get("preferred_doctor_id", "")
        patient = get_patient(patient_id)
        if patient is None:
            return Observation(status="error", message=f"Patient '{patient_id}' not found.", available_actions=ALL_ACTION_NAMES)
        existing = next((w for w in self.waitlist if w.patient_id == patient_id and w.department == department), None)
        if existing:
            return Observation(status="error", message=f"{patient.name} is already on the {department} waitlist.", available_actions=ALL_ACTION_NAMES)
        entry = WaitlistEntry(patient_id=patient_id, department=department, preferred_doctor_id=preferred_doctor_id,
                              date_added="2026-04-01", urgency=patient.urgency, notes=f"Added at step {self.step_number}")
        self.waitlist.append(entry)
        return Observation(status="success", message=f"{patient.name} added to {department} waitlist.",
                           data={"patient_id": patient_id, "department": department}, available_actions=ALL_ACTION_NAMES)

    def _handle_get_doctor_schedule(self, params: dict[str, Any]) -> Observation:
        doctor_id = params.get("doctor_id", "")
        doc = self._get_doctor(doctor_id)
        if doc is None:
            return Observation(status="error", message=f"Doctor '{doctor_id}' not found.", available_actions=ALL_ACTION_NAMES)
        booked = [{"appointment_id": a.appointment_id, "patient_id": a.patient_id, "date": a.slot.date, "start_time": a.slot.start_time, "status": a.status.value}
                   for a in self.appointments if a.doctor_id == doctor_id and a.status in (AppointmentStatus.SCHEDULED, AppointmentStatus.RESCHEDULED)]
        available = [{"slot_id": s.slot_id, "date": s.date, "start_time": s.start_time, "end_time": s.end_time}
                     for s in doc.availability if s.is_available and s.date not in doc.on_leave_dates]
        return Observation(status="success", message=f"Full schedule for {doc.name}.",
                           data={"doctor_id": doctor_id, "doctor_name": doc.name, "department": doc.department.value,
                                 "booked_appointments": booked, "available_slots": available, "on_leave_dates": doc.on_leave_dates,
                                 "max_patients_per_day": doc.max_patients_per_day, "accepted_insurance": doc.accepted_insurance},
                           available_actions=ALL_ACTION_NAMES)

    def _handle_finish(self, params: dict[str, Any]) -> Observation:
        """Agent signals it has completed the task."""
        self.done = True
        return Observation(
            status="success",
            message="Agent signaled task completion.",
            data={"final_reward": self.current_reward},
            available_actions=[],
            done=True,
        )

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _update_reward_for_patient_lookup(self, patient_id: str) -> None:
        """Reward the agent for looking up patient records (symptom-based reasoning)."""
        if not self.task_config:
            return
        targets = set(self.task_config.target_patients)
        if patient_id in targets:
            looked_up_targets = self._patients_looked_up & targets
            fraction = len(looked_up_targets) / max(len(targets), 1)
            self.current_reward = max(self.current_reward, 0.05 + 0.05 * fraction)

    def _update_reward_for_search(self, department: str) -> None:
        if not self.task_config:
            return
        c = self.task_config.constraints
        targets = [c.get("expected_department", ""), c.get("department", ""), c.get("urgent_department", ""), c.get("emergency_department", ""), c.get("routine_department", "")]
        if department.lower().replace(" ", "_") in targets:
            self.current_reward = max(self.current_reward, 0.10)

    def _update_reward_for_availability_check(self) -> None:
        self.current_reward = max(self.current_reward, 0.15)

    def _update_reward_for_detail_retrieval(self) -> None:
        self.current_reward = max(self.current_reward, 0.10)

    def _update_reward_for_insurance_check(self, patient_id: str, coverage: dict) -> None:
        if not self.task_config:
            return
        if self.task_config.constraints.get("insurance_check_required"):
            verified_count = len(self.insurance_verified)
            target_count = len(self.task_config.target_patients)
            self.current_reward = max(self.current_reward, 0.08 + 0.08 * (verified_count / max(target_count, 1)))

    def _update_reward_for_waitlist_check(self) -> None:
        if not self.task_config:
            return
        if self.task_config.constraints.get("waitlist_patient"):
            self.current_reward = max(self.current_reward, 0.12)

    def _update_reward_for_booking(self, patient_id: str, doctor_id: str, slot: TimeSlot) -> None:
        if not self.task_config:
            return
        difficulty = self.task_config.difficulty
        if difficulty == "easy":
            self.current_reward = max(self.current_reward, 0.7)
            constraints = self.task_config.constraints
            dept = constraints.get("expected_department", constraints.get("department", ""))
            date = constraints.get("date", "")
            earliest = self._find_earliest_slot(dept, date)
            if earliest and slot.slot_id == earliest.slot_id:
                self.current_reward = 1.0
            else:
                self.current_reward = max(self.current_reward, 0.8)
        elif difficulty in ("hard", "expert", "nightmare"):
            booked_patients = {a.patient_id for a in self.appointments if a.status == AppointmentStatus.SCHEDULED and a.appointment_id.startswith("APT-5")}
            target = set(self.task_config.target_patients)
            fraction = len(booked_patients & target) / max(len(target), 1)
            base = 0.15 if difficulty == "nightmare" else 0.25
            self.current_reward = max(self.current_reward, base + (1.0 - base) * fraction * 0.85)

    def _update_reward_for_reschedule(self, appt: Appointment, new_slot: TimeSlot) -> None:
        if not self.task_config or self.task_config.difficulty != "medium":
            return
        constraints = self.task_config.constraints
        score = 0.4
        if appt.doctor_id == constraints.get("doctor_id"):
            score += 0.2
        min_time = constraints.get("min_time", "")
        if min_time and new_slot.start_time >= min_time:
            score += 0.2
        start = constraints.get("date_range_start", "")
        end = constraints.get("date_range_end", "")
        if start and end and start <= new_slot.date <= end:
            score += 0.2
        self.current_reward = max(self.current_reward, min(score, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_doctor(self, doctor_id: str) -> Any:
        return next((d for d in self.doctors if d.doctor_id == doctor_id), None)

    def _mark_slot_unavailable(self, slot_id: str) -> None:
        for doc in self.doctors:
            for slot in doc.availability:
                if slot.slot_id == slot_id:
                    slot.is_available = False
                    return

    def _mark_slot_available(self, slot_id: str) -> None:
        for doc in self.doctors:
            for slot in doc.availability:
                if slot.slot_id == slot_id:
                    slot.is_available = True
                    return

    def _find_earliest_slot(self, department: str, date: str) -> TimeSlot | None:
        dept_doctors = [d for d in self.doctors if d.department.value == department.lower().replace(" ", "_")]
        earliest = None
        for doc in dept_doctors:
            for slot in doc.availability:
                if slot.is_available and slot.date == date and slot.date not in doc.on_leave_dates:
                    if earliest is None or slot.start_time < earliest.start_time:
                        earliest = slot
        return earliest

    def _add_next_week_slots(self) -> None:
        doc = self._get_doctor("D003")
        if doc is None:
            return
        for info in [
            ("D003-0406-09", "2026-04-06", "09:00", "09:30"), ("D003-0406-14", "2026-04-06", "14:00", "14:30"),
            ("D003-0407-10", "2026-04-07", "10:00", "10:30"), ("D003-0407-15", "2026-04-07", "15:00", "15:30"),
            ("D003-0408-09", "2026-04-08", "09:00", "09:30"), ("D003-0408-14", "2026-04-08", "14:00", "14:30"),
            ("D003-0409-11", "2026-04-09", "11:00", "11:30"), ("D003-0409-16", "2026-04-09", "16:00", "16:30"),
            ("D003-0410-10", "2026-04-10", "10:00", "10:30"), ("D003-0410-15", "2026-04-10", "15:00", "15:30"),
        ]:
            doc.availability.append(TimeSlot(slot_id=info[0], date=info[1], start_time=info[2], end_time=info[3]))
