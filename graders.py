"""Task-specific graders that compute final scores (0.0 - 1.0).

Graders reward:
- Looking up patient records (symptom-based reasoning)
- Correct department selection based on symptoms
- Insurance verification before booking
- Proper triage ordering
- No double-bookings
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models import ActionType, AppointmentStatus

if TYPE_CHECKING:
    from env import HospitalEnv


def grade(env: HospitalEnv) -> float:
    if env.task_config is None:
        return 0.0
    grader = {
        "task_easy": grade_task_easy,
        "task_medium": grade_task_medium,
        "task_hard": grade_task_hard,
        "task_expert": grade_task_expert,
        "task_nightmare": grade_task_nightmare,
    }.get(env.task_config.task_id)
    if grader is None:
        return env.current_reward
    return grader(env)


def grade_task_easy(env: HospitalEnv) -> float:
    score = 0.0
    actions = env.action_history
    target_patient = (env.task_config.target_patients[0] if env.task_config and env.task_config.target_patients else "P001")

    if any(a.action_type == ActionType.GET_PATIENT_INFO and a.parameters.get("patient_id") == target_patient for a in actions):
        score += 0.15
    if any(a.action_type == ActionType.SEARCH_DOCTORS and a.parameters.get("department", "").lower().replace(" ", "_") == "cardiology" for a in actions):
        score += 0.15
    if any(a.action_type == ActionType.CHECK_AVAILABILITY for a in actions):
        score += 0.10

    booked = [a for a in env.appointments if a.patient_id == target_patient and a.status == AppointmentStatus.SCHEDULED]
    if booked:
        score += 0.40
        if booked[0].slot.start_time == "09:00" and booked[0].slot.date == "2026-04-01":
            score += 0.20

    return min(score, 1.0)


def grade_task_medium(env: HospitalEnv) -> float:
    score = 0.0
    actions = env.action_history
    constraints = env.task_config.constraints if env.task_config else {}

    if any(a.action_type == ActionType.GET_PATIENT_INFO and a.parameters.get("patient_id") == "P002" for a in actions):
        score += 0.10
    if any(a.action_type == ActionType.GET_APPOINTMENT_DETAILS and a.parameters.get("appointment_id") == "APT-101" for a in actions):
        score += 0.10
    if any(a.action_type == ActionType.CHECK_AVAILABILITY and a.parameters.get("doctor_id") == "D003" for a in actions):
        score += 0.05

    rescheduled = next((a for a in env.appointments if a.appointment_id == "APT-101" and a.status == AppointmentStatus.RESCHEDULED), None)
    if rescheduled:
        if rescheduled.doctor_id == constraints.get("doctor_id", "D003"):
            score += 0.25
        if rescheduled.slot.start_time >= constraints.get("min_time", "14:00"):
            score += 0.25
        start = constraints.get("date_range_start", "2026-04-06")
        end = constraints.get("date_range_end", "2026-04-10")
        if start <= rescheduled.slot.date <= end:
            score += 0.25

    return min(score, 1.0)


def grade_task_hard(env: HospitalEnv) -> float:
    score = 0.0
    actions = env.action_history

    target_patients = {"P003", "P004", "P005"}
    looked_up = {a.parameters.get("patient_id") for a in actions if a.action_type == ActionType.GET_PATIENT_INFO}
    score += 0.10 * (len(looked_up & target_patients) / 3.0)

    scheduled = [a for a in env.appointments if a.status == AppointmentStatus.SCHEDULED]

    def find_new(pid):
        return next((a for a in scheduled if a.patient_id == pid and a.appointment_id.startswith("APT-5")), None)

    urgent_appt = find_new("P003")
    pref_appt = find_new("P004")
    seq_appt = find_new("P005")

    if urgent_appt and urgent_appt.slot.date <= "2026-04-02":
        from hospital_data import get_doctor as gd
        doc = gd(urgent_appt.doctor_id)
        if doc and doc.department.value == "neurology":
            score += 0.20
        else:
            score += 0.05

    if pref_appt and pref_appt.doctor_id == "D003":
        score += 0.15
    elif pref_appt:
        from hospital_data import get_doctor as gd2
        doc = gd2(pref_appt.doctor_id)
        if doc and doc.department.value == "orthopedics":
            score += 0.10
        else:
            score += 0.03

    if seq_appt:
        existing = next((a for a in env.appointments if a.appointment_id == "APT-202"), None)
        if existing and (seq_appt.slot.date > existing.slot.date or (seq_appt.slot.date == existing.slot.date and seq_appt.slot.start_time > existing.slot.start_time)):
            from hospital_data import get_doctor as gd3
            doc = gd3(seq_appt.doctor_id)
            if doc and doc.department.value == "cardiology":
                score += 0.15
            else:
                score += 0.05

    booked_slots = [a.slot.slot_id for a in scheduled]
    if len(booked_slots) == len(set(booked_slots)):
        score += 0.15

    booked_new = {a.patient_id for a in scheduled if a.appointment_id.startswith("APT-5")}
    if target_patients.issubset(booked_new):
        score += 0.15
    elif len(booked_new & target_patients) >= 2:
        score += 0.08

    if env.step_number <= 15:
        score += 0.10
    elif env.step_number <= 20:
        score += 0.05

    return min(score, 1.0)


def grade_task_expert(env: HospitalEnv) -> float:
    score = 0.0
    actions = env.action_history

    if any(a.action_type == ActionType.GET_PATIENT_INFO and a.parameters.get("patient_id") == "P004" for a in actions):
        score += 0.05
    if any(a.action_type == ActionType.VERIFY_INSURANCE and a.parameters.get("patient_id") == "P004" for a in actions):
        score += 0.10

    p004_booked = next((a for a in env.appointments if a.patient_id == "P004" and a.status == AppointmentStatus.SCHEDULED and a.appointment_id.startswith("APT-5")), None)
    if p004_booked:
        from hospital_data import get_doctor as gd
        doc = gd(p004_booked.doctor_id)
        if doc and doc.department.value == "general_medicine":
            score += 0.15
        else:
            score += 0.05

    if any(a.action_type == ActionType.GET_PATIENT_INFO and a.parameters.get("patient_id") == "P009" for a in actions):
        score += 0.05
    if any(a.action_type == ActionType.VERIFY_INSURANCE and a.parameters.get("patient_id") == "P009" for a in actions):
        score += 0.10
    if any(a.action_type == ActionType.GET_APPOINTMENT_DETAILS and a.parameters.get("appointment_id") == "APT-404" for a in actions):
        score += 0.10

    p009_booked = next((a for a in env.appointments if a.patient_id == "P009" and a.status == AppointmentStatus.SCHEDULED and a.appointment_id.startswith("APT-5")), None)
    if p009_booked:
        from hospital_data import get_doctor as gd2
        doc = gd2(p009_booked.doctor_id)
        if doc and "GuardianPlan" in doc.accepted_insurance:
            score += 0.15
        else:
            score += 0.05

    booked_new = {a.patient_id for a in env.appointments if a.status == AppointmentStatus.SCHEDULED and a.appointment_id.startswith("APT-5")}
    if {"P004", "P009"}.issubset(booked_new):
        score += 0.15
    elif len(booked_new & {"P004", "P009"}) >= 1:
        score += 0.05

    all_slots = [a.slot.slot_id for a in env.appointments if a.status == AppointmentStatus.SCHEDULED]
    if len(all_slots) == len(set(all_slots)):
        score += 0.05

    if env.step_number <= 15:
        score += 0.05

    return min(score, 1.0)


def grade_task_nightmare(env: HospitalEnv) -> float:
    score = 0.0
    actions = env.action_history
    constraints = env.task_config.constraints if env.task_config else {}
    triage_order = constraints.get("triage_order", ["P008", "P007", "P010", "P006"])
    scheduled = [a for a in env.appointments if a.status == AppointmentStatus.SCHEDULED]

    def find_new(pid):
        return next((a for a in scheduled if a.patient_id == pid and a.appointment_id.startswith("APT-5")), None)

    target_patients = set(triage_order)
    looked_up = {a.parameters.get("patient_id") for a in actions if a.action_type == ActionType.GET_PATIENT_INFO}
    score += 0.05 * (len(looked_up & target_patients) / max(len(target_patients), 1))

    p008_appt = find_new("P008")
    p007_appt = find_new("P007")
    p010_appt = find_new("P010")
    p006_appt = find_new("P006")

    if p008_appt:
        from hospital_data import get_doctor as gd
        doc = gd(p008_appt.doctor_id)
        if doc and doc.department.value == "cardiology" and p008_appt.slot.date <= "2026-04-02":
            score += 0.12
        else:
            score += 0.04
    if any(a.action_type == ActionType.VERIFY_INSURANCE and a.parameters.get("patient_id") == "P008" for a in actions):
        score += 0.05

    if p007_appt:
        from hospital_data import get_doctor as gd2
        doc = gd2(p007_appt.doctor_id)
        if doc and doc.department.value == "pediatrics" and p007_appt.slot.date <= "2026-04-02":
            score += 0.12
        else:
            score += 0.04
    if any(a.action_type == ActionType.VERIFY_INSURANCE and a.parameters.get("patient_id") == "P007" for a in actions):
        score += 0.05

    if any(a.action_type == ActionType.CHECK_WAITLIST for a in actions):
        score += 0.05

    if p010_appt:
        from hospital_data import get_doctor as gd3
        doc = gd3(p010_appt.doctor_id)
        if doc and doc.department.value == "neurology":
            score += 0.10
        else:
            score += 0.03
    if any(a.action_type == ActionType.VERIFY_INSURANCE and a.parameters.get("patient_id") == "P010" for a in actions):
        score += 0.03

    if p006_appt:
        from hospital_data import get_doctor as gd4
        doc = gd4(p006_appt.doctor_id)
        if doc and doc.department.value == "dermatology":
            score += 0.08
        else:
            score += 0.03
    if any(a.action_type == ActionType.VERIFY_INSURANCE and a.parameters.get("patient_id") == "P006" for a in actions):
        score += 0.05

    booked_new = {a.patient_id for a in scheduled if a.appointment_id.startswith("APT-5")}
    if target_patients.issubset(booked_new):
        score += 0.10
    elif len(booked_new & target_patients) >= 3:
        score += 0.06

    all_slots = [a.slot.slot_id for a in scheduled]
    if len(all_slots) == len(set(all_slots)):
        score += 0.05

    booking_order = [a.patient_id for a in env.appointments if a.appointment_id.startswith("APT-5") and a.status == AppointmentStatus.SCHEDULED]
    expected_order = [p for p in triage_order if p in booking_order]
    actual_positions = {pid: i for i, pid in enumerate(booking_order)}
    order_correct = all(actual_positions.get(expected_order[i], 999) < actual_positions.get(expected_order[i + 1], 999) for i in range(len(expected_order) - 1))
    if order_correct and len(expected_order) >= 3:
        score += 0.05

    return min(score, 1.0)
