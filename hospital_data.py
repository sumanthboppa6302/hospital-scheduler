"""Hospital database: insurance plans, doctors, patients, appointments, waitlist."""

from models import (
    Appointment,
    AppointmentStatus,
    Department,
    Doctor,
    InsurancePlan,
    InsuranceStatus,
    Patient,
    TimeSlot,
    UrgencyLevel,
    WaitlistEntry,
)

INSURANCE_PLANS: dict[str, InsurancePlan] = {
    "AzureShield": InsurancePlan(
        plan_name="AzureShield Premium",
        provider="AzureShield",
        status=InsuranceStatus.ACTIVE,
        covered_departments=["cardiology", "orthopedics", "dermatology", "neurology", "general_medicine", "pediatrics"],
        copay=25.0,
        requires_referral=False,
    ),
    "ClearPath": InsurancePlan(
        plan_name="ClearPath Basic",
        provider="ClearPath",
        status=InsuranceStatus.ACTIVE,
        covered_departments=["general_medicine", "orthopedics", "dermatology"],
        copay=40.0,
        requires_referral=True,
        requires_preauth=["orthopedics"],
    ),
    "UnionCare": InsurancePlan(
        plan_name="UnionCare Plus",
        provider="UnionCare",
        status=InsuranceStatus.ACTIVE,
        covered_departments=["cardiology", "neurology", "general_medicine", "emergency", "pediatrics"],
        copay=30.0,
        requires_referral=False,
    ),
    "VeriCare": InsurancePlan(
        plan_name="VeriCare Standard",
        provider="VeriCare",
        status=InsuranceStatus.EXPIRED,
        covered_departments=["orthopedics", "general_medicine"],
        copay=35.0,
        requires_referral=True,
    ),
    "FederalMed": InsurancePlan(
        plan_name="FederalMed Part B",
        provider="FederalMed",
        status=InsuranceStatus.ACTIVE,
        covered_departments=["cardiology", "neurology", "general_medicine", "orthopedics", "emergency"],
        copay=0.0,
        requires_referral=False,
    ),
    "StateWell": InsurancePlan(
        plan_name="StateWell Managed Care",
        provider="StateWell",
        status=InsuranceStatus.ACTIVE,
        covered_departments=["general_medicine", "emergency", "pediatrics"],
        copay=5.0,
        requires_referral=True,
        requires_preauth=["emergency"],
    ),
    "GuardianPlan": InsurancePlan(
        plan_name="GuardianPlan Select",
        provider="GuardianPlan",
        status=InsuranceStatus.ACTIVE,
        covered_departments=["cardiology", "orthopedics", "neurology", "general_medicine", "dermatology", "emergency", "pediatrics"],
        copay=15.0,
        requires_referral=False,
    ),
}

DOCTORS: list[Doctor] = [
    Doctor(
        doctor_id="D001", name="Dr. Anaya Srivastava",
        department=Department.CARDIOLOGY, specialization="Interventional Cardiology",
        rating=4.8, accepted_insurance=["AzureShield", "UnionCare", "FederalMed", "GuardianPlan"],
        max_patients_per_day=8, on_leave_dates=["2026-04-05"],
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D001-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D001-0401-10", date="2026-04-01", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D001-0401-14", date="2026-04-01", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D001-0402-09", date="2026-04-02", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D001-0402-11", date="2026-04-02", start_time="11:00", end_time="11:30"),
            TimeSlot(slot_id="D001-0403-15", date="2026-04-03", start_time="15:00", end_time="15:30"),
            TimeSlot(slot_id="D001-0404-09", date="2026-04-04", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D001-0404-14", date="2026-04-04", start_time="14:00", end_time="14:30"),
        ],
    ),
    Doctor(
        doctor_id="D002", name="Dr. Michael Chen",
        department=Department.CARDIOLOGY, specialization="Electrophysiology",
        rating=4.5, accepted_insurance=["AzureShield", "ClearPath", "FederalMed"],
        max_patients_per_day=6,
        working_hours=("09:00", "17:00"),
        working_days=["Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D002-0401-11", date="2026-04-01", start_time="11:00", end_time="11:30"),
            TimeSlot(slot_id="D002-0401-15", date="2026-04-01", start_time="15:00", end_time="15:30"),
            TimeSlot(slot_id="D002-0402-09", date="2026-04-02", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D002-0403-10", date="2026-04-03", start_time="10:00", end_time="10:30"),
        ],
    ),
    Doctor(
        doctor_id="D003", name="Dr. Rohan Sharma",
        department=Department.ORTHOPEDICS, specialization="Sports Medicine",
        rating=4.7, accepted_insurance=["AzureShield", "ClearPath", "VeriCare", "FederalMed", "GuardianPlan"],
        max_patients_per_day=8,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D003-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D003-0401-14", date="2026-04-01", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D003-0402-10", date="2026-04-02", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D003-0402-15", date="2026-04-02", start_time="15:00", end_time="15:30"),
            TimeSlot(slot_id="D003-0403-09", date="2026-04-03", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D003-0404-11", date="2026-04-04", start_time="11:00", end_time="11:30"),
        ],
    ),
    Doctor(
        doctor_id="D004", name="Dr. Emily Davis",
        department=Department.ORTHOPEDICS, specialization="Joint Replacement",
        rating=4.3, accepted_insurance=["AzureShield", "FederalMed", "UnionCare"],
        max_patients_per_day=6,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D004-0401-10", date="2026-04-01", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D004-0402-14", date="2026-04-02", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D004-0403-09", date="2026-04-03", start_time="09:00", end_time="09:30"),
        ],
    ),
    Doctor(
        doctor_id="D005", name="Dr. Lisa Wang",
        department=Department.DERMATOLOGY, specialization="Clinical Dermatology",
        rating=4.6, accepted_insurance=["AzureShield", "ClearPath", "GuardianPlan"],
        max_patients_per_day=10,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D005-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D005-0401-11", date="2026-04-01", start_time="11:00", end_time="11:30"),
            TimeSlot(slot_id="D005-0402-14", date="2026-04-02", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D005-0403-10", date="2026-04-03", start_time="10:00", end_time="10:30"),
        ],
    ),
    Doctor(
        doctor_id="D006", name="Dr. Marcus Leong",
        department=Department.DERMATOLOGY, specialization="Cosmetic Dermatology",
        rating=4.2, accepted_insurance=["AzureShield", "GuardianPlan"],
        max_patients_per_day=8,
        working_hours=("10:00", "18:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D006-0401-14", date="2026-04-01", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D006-0402-09", date="2026-04-02", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D006-0403-15", date="2026-04-03", start_time="15:00", end_time="15:30"),
        ],
    ),
    Doctor(
        doctor_id="D007", name="Dr. Amanda Foster",
        department=Department.NEUROLOGY, specialization="Neurophysiology",
        rating=4.9, accepted_insurance=["AzureShield", "UnionCare", "FederalMed", "GuardianPlan"],
        max_patients_per_day=6,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D007-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D007-0401-14", date="2026-04-01", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D007-0402-10", date="2026-04-02", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D007-0402-15", date="2026-04-02", start_time="15:00", end_time="15:30"),
            TimeSlot(slot_id="D007-0403-09", date="2026-04-03", start_time="09:00", end_time="09:30"),
        ],
    ),
    Doctor(
        doctor_id="D008", name="Dr. Kevin Park",
        department=Department.NEUROLOGY, specialization="Stroke & Cerebrovascular",
        rating=4.4, accepted_insurance=["UnionCare", "FederalMed", "StateWell"],
        max_patients_per_day=6,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D008-0401-10", date="2026-04-01", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D008-0402-11", date="2026-04-02", start_time="11:00", end_time="11:30"),
            TimeSlot(slot_id="D008-0403-14", date="2026-04-03", start_time="14:00", end_time="14:30"),
        ],
    ),
    Doctor(
        doctor_id="D009", name="Dr. Priya Reddy",
        department=Department.GENERAL_MEDICINE, specialization="Internal Medicine",
        rating=4.6, accepted_insurance=["AzureShield", "ClearPath", "UnionCare", "VeriCare", "FederalMed", "StateWell", "GuardianPlan"],
        max_patients_per_day=12,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D009-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D009-0401-11", date="2026-04-01", start_time="11:00", end_time="11:30"),
            TimeSlot(slot_id="D009-0401-14", date="2026-04-01", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D009-0402-10", date="2026-04-02", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D009-0403-09", date="2026-04-03", start_time="09:00", end_time="09:30"),
        ],
    ),
    Doctor(
        doctor_id="D010", name="Dr. Thomas Lee",
        department=Department.GENERAL_MEDICINE, specialization="Family Medicine",
        rating=4.1, accepted_insurance=["AzureShield", "ClearPath", "VeriCare", "StateWell"],
        max_patients_per_day=10,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D010-0401-10", date="2026-04-01", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D010-0402-14", date="2026-04-02", start_time="14:00", end_time="14:30"),
            TimeSlot(slot_id="D010-0403-11", date="2026-04-03", start_time="11:00", end_time="11:30"),
        ],
    ),
    Doctor(
        doctor_id="D011", name="Dr. Rachel Torres",
        department=Department.EMERGENCY, specialization="Emergency Medicine",
        rating=4.7, accepted_insurance=["AzureShield", "UnionCare", "FederalMed", "StateWell", "GuardianPlan"],
        max_patients_per_day=20,
        working_hours=("08:00", "20:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        availability=[
            TimeSlot(slot_id="D011-0401-08", date="2026-04-01", start_time="08:00", end_time="08:30"),
            TimeSlot(slot_id="D011-0401-12", date="2026-04-01", start_time="12:00", end_time="12:30"),
            TimeSlot(slot_id="D011-0401-16", date="2026-04-01", start_time="16:00", end_time="16:30"),
            TimeSlot(slot_id="D011-0402-08", date="2026-04-02", start_time="08:00", end_time="08:30"),
            TimeSlot(slot_id="D011-0402-14", date="2026-04-02", start_time="14:00", end_time="14:30"),
        ],
    ),
    Doctor(
        doctor_id="D012", name="Dr. Nancy Patel",
        department=Department.PEDIATRICS, specialization="General Pediatrics",
        rating=4.8, accepted_insurance=["AzureShield", "UnionCare", "StateWell", "GuardianPlan"],
        max_patients_per_day=10,
        working_hours=("09:00", "17:00"),
        working_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        availability=[
            TimeSlot(slot_id="D012-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30"),
            TimeSlot(slot_id="D012-0401-11", date="2026-04-01", start_time="11:00", end_time="11:30"),
            TimeSlot(slot_id="D012-0402-10", date="2026-04-02", start_time="10:00", end_time="10:30"),
            TimeSlot(slot_id="D012-0403-14", date="2026-04-03", start_time="14:00", end_time="14:30"),
        ],
    ),
]

PATIENTS: list[Patient] = [
    Patient(
        patient_id="P001", name="Marcus Webb", age=45,
        symptoms=["chest pain", "shortness of breath"],
        insurance="AzureShield", phone="555-0101",
        urgency=UrgencyLevel.SOON,
        allergies=["penicillin"],
        medical_history=["hypertension", "high cholesterol"],
    ),
    Patient(
        patient_id="P002", name="Jane Martinez", age=32,
        symptoms=["knee pain", "swelling"],
        insurance="ClearPath", phone="555-0102",
        urgency=UrgencyLevel.ROUTINE,
        allergies=[],
        medical_history=["ACL tear (2024)"],
    ),
    Patient(
        patient_id="P003", name="David Kim", age=58,
        symptoms=["severe headache", "dizziness", "numbness in left arm"],
        insurance="UnionCare", phone="555-0103",
        urgency=UrgencyLevel.URGENT,
        allergies=["sulfa drugs"],
        medical_history=["type 2 diabetes", "previous TIA"],
    ),
    Patient(
        patient_id="P004", name="Sarah Thompson", age=28,
        symptoms=["back pain", "stiffness"],
        insurance="VeriCare", phone="555-0104",
        urgency=UrgencyLevel.ROUTINE,
        allergies=[],
        medical_history=[],
    ),
    Patient(
        patient_id="P005", name="Michael Rivera", age=63,
        symptoms=["irregular heartbeat", "fatigue", "chest tightness"],
        insurance="FederalMed", phone="555-0105",
        urgency=UrgencyLevel.SOON,
        allergies=["aspirin"],
        medical_history=["atrial fibrillation", "coronary artery disease"],
    ),
    Patient(
        patient_id="P006", name="Emily Chang", age=41,
        symptoms=["skin rash", "itching", "spreading lesions"],
        insurance="AzureShield", phone="555-0106",
        urgency=UrgencyLevel.ROUTINE,
        allergies=["latex"],
        medical_history=["eczema"],
    ),
    Patient(
        patient_id="P007", name="Tommy Nakamoto", age=7,
        symptoms=["high fever", "ear pain", "irritability"],
        insurance="StateWell", phone="555-0107",
        urgency=UrgencyLevel.URGENT,
        allergies=["amoxicillin"],
        medical_history=["recurrent ear infections"],
    ),
    Patient(
        patient_id="P008", name="Robert Hayes", age=72,
        symptoms=["chest pain", "radiating to left arm", "nausea"],
        insurance="FederalMed", phone="555-0108",
        urgency=UrgencyLevel.EMERGENCY,
        allergies=[],
        medical_history=["previous MI (2023)", "hypertension", "diabetes"],
    ),
    Patient(
        patient_id="P009", name="Lisa Nakamura", age=35,
        symptoms=["persistent cough", "fatigue", "weight loss"],
        insurance="GuardianPlan", phone="555-0109",
        urgency=UrgencyLevel.SOON,
        allergies=[],
        medical_history=[],
    ),
    Patient(
        patient_id="P010", name="Carlos Mendez", age=50,
        symptoms=["numbness in feet", "blurred vision"],
        insurance="UnionCare", phone="555-0110",
        urgency=UrgencyLevel.SOON,
        allergies=["ibuprofen"],
        medical_history=["type 2 diabetes", "diabetic retinopathy"],
    ),
]

SEED_APPOINTMENTS: list[Appointment] = [
    Appointment(
        appointment_id="APT-101",
        patient_id="P002", doctor_id="D003",
        slot=TimeSlot(slot_id="D003-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30", is_available=False),
        status=AppointmentStatus.SCHEDULED,
        notes="Initial orthopedics consultation",
        urgency=UrgencyLevel.ROUTINE,
    ),
    Appointment(
        appointment_id="APT-202",
        patient_id="P005", doctor_id="D001",
        slot=TimeSlot(slot_id="D001-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30", is_available=False),
        status=AppointmentStatus.SCHEDULED,
        notes="Cardiology follow-up",
        urgency=UrgencyLevel.SOON,
    ),
    Appointment(
        appointment_id="APT-303",
        patient_id="P008", doctor_id="D011",
        slot=TimeSlot(slot_id="D011-0401-08", date="2026-04-01", start_time="08:00", end_time="08:30", is_available=False),
        status=AppointmentStatus.SCHEDULED,
        notes="Emergency admission -- chest pain with cardiac history",
        urgency=UrgencyLevel.EMERGENCY,
    ),
    Appointment(
        appointment_id="APT-404",
        patient_id="P009", doctor_id="D009",
        slot=TimeSlot(slot_id="D009-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30", is_available=False),
        status=AppointmentStatus.SCHEDULED,
        notes="General checkup -- referred for specialist follow-up",
        urgency=UrgencyLevel.SOON,
    ),
    Appointment(
        appointment_id="APT-505",
        patient_id="P002", doctor_id="D007",
        slot=TimeSlot(slot_id="D007-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30", is_available=False),
        status=AppointmentStatus.SCHEDULED,
        notes="Neurology consult",
        urgency=UrgencyLevel.ROUTINE,
    ),
    Appointment(
        appointment_id="APT-606",
        patient_id="P005", doctor_id="D012",
        slot=TimeSlot(slot_id="D012-0401-09", date="2026-04-01", start_time="09:00", end_time="09:30", is_available=False),
        status=AppointmentStatus.SCHEDULED,
        notes="Pediatrics well-visit",
        urgency=UrgencyLevel.ROUTINE,
    ),
]

SEED_WAITLIST: list[WaitlistEntry] = [
    WaitlistEntry(
        patient_id="P010", department="neurology",
        preferred_doctor_id="D007", date_added="2026-03-28",
        urgency=UrgencyLevel.SOON,
        notes="Waiting for neurology opening with Dr. Amanda Foster",
    ),
]


def get_doctor(doctor_id: str) -> Doctor | None:
    return next((d for d in DOCTORS if d.doctor_id == doctor_id), None)


def get_patient(patient_id: str) -> Patient | None:
    return next((p for p in PATIENTS if p.patient_id == patient_id), None)


def get_doctors_by_department(department: str) -> list[Doctor]:
    dept = department.lower().replace(" ", "_")
    return [d for d in DOCTORS if d.department.value == dept]


def get_insurance_plan(provider: str) -> InsurancePlan | None:
    return INSURANCE_PLANS.get(provider)


def check_insurance_coverage(patient_id: str, department: str) -> dict:
    patient = get_patient(patient_id)
    if not patient:
        return {"covered": False, "reason": "Patient not found"}

    plan = get_insurance_plan(patient.insurance)
    if not plan:
        return {"covered": False, "reason": f"Unknown insurance provider: {patient.insurance}"}

    if plan.status == InsuranceStatus.EXPIRED:
        return {
            "covered": False,
            "reason": f"Insurance plan '{plan.plan_name}' has expired",
            "plan_status": "expired",
            "provider": plan.provider,
        }

    if plan.status == InsuranceStatus.PENDING:
        return {
            "covered": False,
            "reason": f"Insurance plan '{plan.plan_name}' is pending activation",
            "plan_status": "pending",
            "provider": plan.provider,
        }

    dept = department.lower().replace(" ", "_")
    if dept not in plan.covered_departments:
        return {
            "covered": False,
            "reason": f"Department '{department}' is not covered by {plan.plan_name}",
            "plan_status": "active",
            "covered_departments": plan.covered_departments,
            "provider": plan.provider,
        }

    return {
        "covered": True,
        "plan_name": plan.plan_name,
        "provider": plan.provider,
        "copay": plan.copay,
        "requires_referral": plan.requires_referral,
        "requires_preauth": plan.requires_preauth,
        "plan_status": "active",
    }
