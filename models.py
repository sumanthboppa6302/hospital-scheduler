"""Typed Pydantic models for the Hospital Appointment OpenEnv environment.

Action and Observation inherit from openenv base types for spec compliance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Department(str, Enum):
    CARDIOLOGY = "cardiology"
    ORTHOPEDICS = "orthopedics"
    DERMATOLOGY = "dermatology"
    NEUROLOGY = "neurology"
    GENERAL_MEDICINE = "general_medicine"
    EMERGENCY = "emergency"
    PEDIATRICS = "pediatrics"


class ActionType(str, Enum):
    SEARCH_DOCTORS = "search_doctors"
    CHECK_AVAILABILITY = "check_availability"
    BOOK_APPOINTMENT = "book_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    GET_PATIENT_INFO = "get_patient_info"
    LIST_DEPARTMENTS = "list_departments"
    GET_APPOINTMENT_DETAILS = "get_appointment_details"
    VERIFY_INSURANCE = "verify_insurance"
    CHECK_WAITLIST = "check_waitlist"
    ADD_TO_WAITLIST = "add_to_waitlist"
    GET_DOCTOR_SCHEDULE = "get_doctor_schedule"
    FINISH = "finish"


class AppointmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    RESCHEDULED = "rescheduled"
    WAITLISTED = "waitlisted"


class UrgencyLevel(str, Enum):
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class InsuranceStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    PENDING = "pending"
    NOT_COVERED = "not_covered"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class TimeSlot(BaseModel):
    slot_id: str
    date: str
    start_time: str
    end_time: str
    is_available: bool = True


class Doctor(BaseModel):
    doctor_id: str
    name: str
    department: Department
    specialization: str
    rating: float = Field(ge=0.0, le=5.0)
    availability: list[TimeSlot] = []
    accepted_insurance: list[str] = []
    max_patients_per_day: int = 8
    on_leave_dates: list[str] = []


class Patient(BaseModel):
    patient_id: str
    name: str
    age: int
    symptoms: list[str] = []
    insurance: str = ""
    phone: str = ""
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    allergies: list[str] = []
    medical_history: list[str] = []


class InsurancePlan(BaseModel):
    plan_name: str
    provider: str
    status: InsuranceStatus = InsuranceStatus.ACTIVE
    covered_departments: list[str] = []
    copay: float = 0.0
    requires_referral: bool = False


class WaitlistEntry(BaseModel):
    patient_id: str
    department: str
    preferred_doctor_id: str = ""
    date_added: str = ""
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    notes: str = ""


class Appointment(BaseModel):
    appointment_id: str
    patient_id: str
    doctor_id: str
    slot: TimeSlot
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    notes: str = ""
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    insurance_verified: bool = False


# ---------------------------------------------------------------------------
# OpenEnv-compliant Action / Observation / State
# ---------------------------------------------------------------------------

class HospitalAction(OpenEnvAction):
    """Action the agent sends to the hospital environment."""
    action_type: str = Field(..., description="One of: search_doctors, check_availability, book_appointment, cancel_appointment, reschedule_appointment, get_patient_info, list_departments, get_appointment_details, verify_insurance, check_waitlist, add_to_waitlist, get_doctor_schedule")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters as key-value pairs")


class HospitalObservation(OpenEnvObservation):
    """Observation returned by the hospital environment after each step."""
    status: str = Field(default="success", description="success | error | warning")
    message: str = Field(default="", description="Human-readable result message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured result data")
    available_actions: list[str] = Field(default_factory=list, description="Available action types")
    step_number: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=0, description="Maximum steps for this episode")


class HospitalState(OpenEnvState):
    """Full snapshot of the hospital environment state."""
    task_id: str = ""
    task_prompt: str = ""
    max_steps: int = 0
    done_flag: bool = False
    current_reward: float = 0.0
    appointments: list[dict] = Field(default_factory=list)
    waitlist: list[dict] = Field(default_factory=list)
    action_history: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task config (internal, not part of OpenEnv interface)
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    task_id: str
    prompt: str
    max_steps: int = 20
    difficulty: str = "easy"
    initial_appointments: list[str] = []
    target_patients: list[str] = []
    constraints: dict[str, Any] = {}
