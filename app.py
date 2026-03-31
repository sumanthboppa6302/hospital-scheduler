"""Gradio UI dashboard for the Hospital Appointment Scheduler OpenEnv.

Two modes:
  1. Manual mode  — you pick actions yourself (debugging)
  2. AI Agent mode — Claude autonomously solves the task (real agentic flow)
"""

from __future__ import annotations

import json
import os
import traceback

from dotenv import load_dotenv
import gradio as gr

load_dotenv()

from env import HospitalEnv, Action
from graders import grade
from models import ActionType

# Global environment instance for UI
ui_env = HospitalEnv()

TASK_CHOICES = [
    ("Easy: Simple Appointment Booking", "task_easy"),
    ("Medium: Reschedule with Constraints", "task_medium"),
    ("Hard: Multi-Patient Scheduling", "task_hard"),
    ("Expert: Insurance-Aware Scheduling", "task_expert"),
    ("Nightmare: Emergency Triage", "task_nightmare"),
]

ACTION_TEMPLATES = {
    "search_doctors": '{"department": "cardiology", "date": "2026-04-01"}',
    "check_availability": '{"doctor_id": "D001", "date": "2026-04-01"}',
    "book_appointment": '{"doctor_id": "D001", "patient_id": "P001", "slot_id": "D001-0401-09"}',
    "cancel_appointment": '{"appointment_id": "APT-500"}',
    "reschedule_appointment": '{"appointment_id": "APT-101", "new_slot_id": "D003-0406-14"}',
    "get_patient_info": '{"patient_id": "P001"}',
    "list_departments": '{}',
    "get_appointment_details": '{"appointment_id": "APT-101"}',
    "verify_insurance": '{"patient_id": "P001", "department": "cardiology"}',
    "check_waitlist": '{"department": "neurology"}',
    "add_to_waitlist": '{"patient_id": "P001", "department": "cardiology"}',
    "get_doctor_schedule": '{"doctor_id": "D001"}',
    "finish": '{}',
}

HEURISTIC_SEQUENCES = {
    "task_easy": [
        ("get_patient_info", {"patient_id": "P001"}),
        ("search_doctors", {"department": "cardiology", "date": "2026-04-01"}),
        ("check_availability", {"doctor_id": "D001", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D001", "patient_id": "P001", "slot_id": "D001-0401-09"}),
    ],
    "task_medium": [
        ("get_patient_info", {"patient_id": "P002"}),
        ("get_appointment_details", {"appointment_id": "APT-101"}),
        ("check_availability", {"doctor_id": "D003", "date": "2026-04-06"}),
        ("reschedule_appointment", {"appointment_id": "APT-101", "new_slot_id": "D003-0406-14"}),
    ],
    "task_hard": [
        ("get_patient_info", {"patient_id": "P003"}),
        ("search_doctors", {"department": "neurology", "date": "2026-04-01"}),
        ("check_availability", {"doctor_id": "D007", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D007", "patient_id": "P003", "slot_id": "D007-0401-09"}),
        ("get_patient_info", {"patient_id": "P004"}),
        ("search_doctors", {"department": "orthopedics", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D003", "patient_id": "P004", "slot_id": "D003-0401-14"}),
        ("get_patient_info", {"patient_id": "P005"}),
        ("get_appointment_details", {"appointment_id": "APT-202"}),
        ("search_doctors", {"department": "cardiology", "date": "2026-04-02"}),
        ("book_appointment", {"doctor_id": "D001", "patient_id": "P005", "slot_id": "D001-0402-09"}),
    ],
    "task_expert": [
        ("get_patient_info", {"patient_id": "P004"}),
        ("verify_insurance", {"patient_id": "P004", "department": "orthopedics"}),
        ("verify_insurance", {"patient_id": "P004", "department": "general_medicine"}),
        ("search_doctors", {"department": "general_medicine", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D009", "patient_id": "P004", "slot_id": "D009-0401-11"}),
        ("get_patient_info", {"patient_id": "P009"}),
        ("get_appointment_details", {"appointment_id": "APT-404"}),
        ("verify_insurance", {"patient_id": "P009", "department": "general_medicine"}),
        ("search_doctors", {"department": "general_medicine", "date": "2026-04-02"}),
        ("book_appointment", {"doctor_id": "D009", "patient_id": "P009", "slot_id": "D009-0402-10"}),
    ],
    "task_nightmare": [
        ("get_patient_info", {"patient_id": "P008"}),
        ("verify_insurance", {"patient_id": "P008", "department": "cardiology"}),
        ("get_appointment_details", {"appointment_id": "APT-303"}),
        ("search_doctors", {"department": "cardiology", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D001", "patient_id": "P008", "slot_id": "D001-0401-10", "urgency": "emergency"}),
        ("get_patient_info", {"patient_id": "P007"}),
        ("verify_insurance", {"patient_id": "P007", "department": "pediatrics"}),
        ("search_doctors", {"department": "pediatrics", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D012", "patient_id": "P007", "slot_id": "D012-0401-11", "urgency": "urgent"}),
        ("get_patient_info", {"patient_id": "P010"}),
        ("check_waitlist", {"department": "neurology"}),
        ("verify_insurance", {"patient_id": "P010", "department": "neurology"}),
        ("search_doctors", {"department": "neurology", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D007", "patient_id": "P010", "slot_id": "D007-0401-14"}),
        ("get_patient_info", {"patient_id": "P006"}),
        ("verify_insurance", {"patient_id": "P006", "department": "dermatology"}),
        ("search_doctors", {"department": "dermatology", "date": "2026-04-01"}),
        ("book_appointment", {"doctor_id": "D005", "patient_id": "P006", "slot_id": "D005-0401-09"}),
    ],
}

SYSTEM_PROMPT = """You are an AI hospital scheduling agent. You interact with a hospital environment by choosing actions.

Available actions and their parameters:
- search_doctors: {"department": "...", "date": "YYYY-MM-DD"} - Find doctors in a department
- check_availability: {"doctor_id": "...", "date": "YYYY-MM-DD"} - Check a doctor's open slots
- book_appointment: {"doctor_id": "...", "patient_id": "...", "slot_id": "...", "urgency": "routine|urgent|emergency"} - Book an appointment
- cancel_appointment: {"appointment_id": "..."} - Cancel appointment
- reschedule_appointment: {"appointment_id": "...", "new_slot_id": "..."} - Reschedule
- get_patient_info: {"patient_id": "..."} - Get patient details, history, allergies
- list_departments: {} - List all departments
- get_appointment_details: {"appointment_id": "..."} - Get appointment info
- verify_insurance: {"patient_id": "...", "department": "..."} - Check insurance coverage
- check_waitlist: {"department": "...", "patient_id": "..."} - View waitlist
- add_to_waitlist: {"patient_id": "...", "department": "..."} - Add to waitlist
- get_doctor_schedule: {"doctor_id": "..."} - Full doctor schedule
- finish: {} - Signal that you have completed all tasks. Call this when done.

IMPORTANT RULES:
- Always verify insurance before booking when dealing with insurance-related tasks
- Always check_availability before booking an appointment
- Triage patients by urgency: emergency > urgent > soon > routine
- Check the waitlist before searching for openings if a patient is waitlisted
- Avoid double-booking slots
- When all tasks are complete, call "finish" immediately. Do NOT keep making unnecessary actions.

Respond with a JSON object with two fields:
1. "reasoning": A brief explanation of your thinking (1-2 sentences)
2. "action": {"action_type": "...", "parameters": {...}}

Example:
{"reasoning": "I need to find cardiologists available on April 1st for the patient.", "action": {"action_type": "search_doctors", "parameters": {"department": "cardiology", "date": "2026-04-01"}}}"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def format_status(state):
    if not state.task_id:
        return "No task loaded. Select a task and click Reset."
    reward_bar_len = int(state.current_reward * 20)
    reward_bar = "#" * reward_bar_len + "-" * (20 - reward_bar_len)
    return "\n".join([
        f"Task:     {state.task_id} ({'DONE' if state.done else 'ACTIVE'})",
        f"Steps:    {state.step_number} / {state.max_steps}",
        f"Reward:   [{reward_bar}] {state.current_reward:.2f}",
        f"Appts:    {len([a for a in state.appointments if a.status.value == 'scheduled'])} scheduled",
        f"Waitlist: {len(state.waitlist)} entries",
    ])


def format_appointments(state):
    if not state.appointments:
        return "No appointments yet."
    lines = []
    for a in state.appointments:
        lines.append(f"  {a.appointment_id} | {a.patient_id} | {a.doctor_id} | {a.slot.date} {a.slot.start_time} | {a.status.value} | ins_verified:{a.insurance_verified}")
    return "\n".join(lines)


def status_icon(status: str) -> str:
    if status == "success":
        return "[OK]"
    if status == "warning":
        return "[!!]"
    return "[ERR]"


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------

def reset_env(task_id: str):
    obs = ui_env.reset(task_id)
    state = ui_env.state()
    log = f"--- RESET: {task_id} ---\n{obs.message}\n"
    return log, format_status(state), format_appointments(state), ""


def take_action(action_type: str, params_json: str, log: str):
    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError:
        return log + "\n[ERROR] Invalid JSON parameters.\n", format_status(ui_env.state()), format_appointments(ui_env.state()), ""

    action = Action(action_type=ActionType(action_type), parameters=params)
    obs = ui_env.step(action)

    log += f"\nStep {obs.step_number}: {action_type} {status_icon(obs.status)}\n  {obs.message}\n"
    if obs.data:
        log += f"  Data: {json.dumps(obs.data, indent=2)[:500]}\n"

    state = ui_env.state()
    if obs.done:
        final_score = grade(ui_env)
        log += f"\n{'='*50}\n  EPISODE DONE - Final Score: {final_score:.2f} / 1.00\n{'='*50}\n"

    return log, format_status(state), format_appointments(state), ""


# ---------------------------------------------------------------------------
# Heuristic agent (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def run_heuristic_agent(task_id: str):
    """Runs the heuristic agent and yields log updates step by step."""
    env = HospitalEnv()
    obs = env.reset(task_id)
    log = f"{'='*55}\n  HEURISTIC AGENT - {task_id}\n{'='*55}\n"
    log += f"Task: {obs.message[:200]}...\n\n"

    steps = HEURISTIC_SEQUENCES.get(task_id, [])
    if not steps:
        yield log + "[ERROR] No heuristic sequence for this task.\n", "", ""
        return

    for i, (atype, params) in enumerate(steps, 1):
        action = Action(action_type=ActionType(atype), parameters=params)
        obs = env.step(action)
        icon = status_icon(obs.status)

        log += f"Step {i}: {atype} {icon}\n"
        log += f"  {obs.message}\n"
        if obs.data:
            log += f"  Data: {json.dumps(obs.data, indent=2)[:300]}\n"
        log += "\n"

        yield log, format_status(env.state()), format_appointments(env.state())

        if obs.done:
            break

    final_score = grade(env)
    log += f"{'='*55}\n"
    log += f"  FINAL SCORE: {final_score:.2f} / 1.00\n"
    log += f"  Steps used: {env.step_number} / {env.task_config.max_steps}\n"
    log += f"{'='*55}\n"
    yield log, format_status(env.state()), format_appointments(env.state())


# ---------------------------------------------------------------------------
# LLM Agent (Groq) -- real agentic flow
# ---------------------------------------------------------------------------

def run_llm_agent(task_id: str, api_key: str):
    """Runs Groq LLM as an autonomous agent. Yields log updates after each step."""
    if not api_key.strip():
        yield "ERROR: Please enter your Groq API key.\n", "", ""
        return

    try:
        from groq import Groq
    except ImportError:
        yield "ERROR: 'groq' package not installed. Run: pip install groq\n", "", ""
        return

    client = Groq(api_key=api_key.strip())
    env = HospitalEnv()
    obs = env.reset(task_id)

    log = f"{'='*55}\n  LLM AGENT (Groq) - {task_id}\n{'='*55}\n"
    log += f"Task: {obs.message[:300]}...\n\n"
    yield log, format_status(env.state()), format_appointments(env.state())

    messages = [
        {"role": "user", "content": f"Here is your task:\n\n{obs.message}\n\nDecide your first action."}
    ]

    max_steps = obs.max_steps
    for step_i in range(max_steps):
        log += f"--- Agent thinking (step {step_i + 1}) ---\n"
        yield log, format_status(env.state()), format_appointments(env.state())

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=600,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            log += f"[API ERROR] {e}\n"
            yield log, format_status(env.state()), format_appointments(env.state())
            return

        # Parse response
        reasoning = ""
        action_data = None
        try:
            # Try to parse the full JSON with reasoning + action
            parsed = json.loads(text)
            reasoning = parsed.get("reasoning", "")
            action_data = parsed.get("action", parsed)
            if "action_type" not in action_data and "action" in parsed:
                action_data = parsed["action"]
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    parsed = json.loads(match.group())
                    reasoning = parsed.get("reasoning", "")
                    action_data = parsed.get("action", parsed)
                except json.JSONDecodeError:
                    pass

        if not action_data or "action_type" not in action_data:
            log += f"  [Agent could not decide an action]\n  Raw: {text[:200]}\n\n"
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Please respond with valid JSON: {\"reasoning\": \"...\", \"action\": {\"action_type\": \"...\", \"parameters\": {...}}}"})
            yield log, format_status(env.state()), format_appointments(env.state())
            continue

        # Show reasoning
        if reasoning:
            log += f"  Thinking: {reasoning}\n"

        # Execute action
        action = Action(action_type=ActionType(action_data["action_type"]), parameters=action_data.get("parameters", {}))
        obs = env.step(action)
        icon = status_icon(obs.status)

        log += f"  Action:  {action.action_type.value} {json.dumps(action.parameters)}\n"
        log += f"  Result:  {icon} {obs.message}\n"
        if obs.data:
            data_preview = json.dumps(obs.data, indent=2)[:400]
            log += f"  Data:    {data_preview}\n"
        log += f"  Reward:  {obs.reward:.2f}\n\n"

        yield log, format_status(env.state()), format_appointments(env.state())

        if obs.done:
            break

        # Feed result back to Claude
        messages.append({"role": "assistant", "content": text})
        feedback = f"Result: {obs.message}\nData: {json.dumps(obs.data)}\nReward so far: {obs.reward:.2f}\nSteps remaining: {obs.max_steps - obs.step_number}\n\nDecide your next action."
        messages.append({"role": "user", "content": feedback})

    final_score = grade(env)
    log += f"{'='*55}\n"
    log += f"  FINAL SCORE: {final_score:.2f} / 1.00\n"
    log += f"  Steps used: {env.step_number} / {env.task_config.max_steps}\n"
    log += f"{'='*55}\n"
    yield log, format_status(env.state()), format_appointments(env.state())


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Hospital OpenEnv") as demo:
    gr.Markdown("# Hospital Appointment Scheduler -- OpenEnv")
    gr.Markdown("AI agent evaluation environment for hospital scheduling with insurance, triage, and waitlist management.")

    with gr.Tabs():
        # ============================================================
        # TAB 1: AI AGENT (the real agentic flow)
        # ============================================================
        with gr.Tab("AI Agent (Autonomous)"):
            gr.Markdown("### Watch an AI agent autonomously solve scheduling tasks")
            gr.Markdown("The agent reads the task, **reasons about what to do**, picks an action, reads the result, and repeats -- fully autonomous. No human in the loop.")

            with gr.Row():
                with gr.Column(scale=1):
                    agent_task = gr.Dropdown(choices=TASK_CHOICES, label="Select Task", value="task_easy")
                    agent_type = gr.Radio(
                        choices=["Heuristic (no API key needed)", "Groq LLM (needs API key)"],
                        label="Agent Type",
                        value="Heuristic (no API key needed)",
                    )
                    api_key_input = gr.Textbox(label="Groq API Key (for LLM agent)", type="password", placeholder="gsk_...", visible=True)
                    run_agent_btn = gr.Button("Run Agent", variant="primary", size="lg")

                    gr.Markdown("### Live Status")
                    agent_status = gr.Textbox(label="Environment State", lines=5, interactive=False, value="Select a task and click Run Agent.")
                    agent_appts = gr.Textbox(label="Appointments", lines=6, interactive=False, value="No appointments yet.")

                with gr.Column(scale=2):
                    gr.Markdown("### Agent Reasoning + Action Log")
                    agent_log = gr.Textbox(label="Live Agent Log", lines=30, interactive=False, max_lines=50, value="Waiting for agent to start...")

            def dispatch_agent(task_id, agent_type_val, api_key):
                if "Heuristic" in agent_type_val:
                    yield from run_heuristic_agent(task_id)
                else:
                    yield from run_llm_agent(task_id, api_key)

            run_agent_btn.click(
                fn=dispatch_agent,
                inputs=[agent_task, agent_type, api_key_input],
                outputs=[agent_log, agent_status, agent_appts],
            )

        # ============================================================
        # TAB 2: MANUAL MODE (debugging)
        # ============================================================
        with gr.Tab("Manual Mode (Debug)"):
            gr.Markdown("### Step through actions manually -- useful for debugging")

            with gr.Row():
                with gr.Column(scale=1):
                    manual_task = gr.Dropdown(choices=TASK_CHOICES, label="Select Task", value="task_easy")
                    manual_reset_btn = gr.Button("Reset Environment", variant="primary")

                    gr.Markdown("#### Pick an Action")
                    manual_action = gr.Dropdown(choices=[a.value for a in ActionType], label="Action Type", value="search_doctors")
                    manual_params = gr.Textbox(label="Parameters (JSON)", value='{"department": "cardiology", "date": "2026-04-01"}', lines=3)
                    manual_step_btn = gr.Button("Execute Action", variant="secondary")

                with gr.Column(scale=2):
                    manual_status = gr.Textbox(label="Status", value="No task loaded.", lines=5, interactive=False)
                    manual_appts = gr.Textbox(label="Appointments", value="No appointments yet.", lines=6, interactive=False)
                    manual_log = gr.Textbox(label="Log", value="", lines=18, interactive=False, max_lines=30)

            manual_action.change(fn=lambda a: ACTION_TEMPLATES.get(a, "{}"), inputs=[manual_action], outputs=[manual_params])

            manual_reset_btn.click(
                fn=reset_env,
                inputs=[manual_task],
                outputs=[manual_log, manual_status, manual_appts, manual_params],
            )

            manual_step_btn.click(
                fn=take_action,
                inputs=[manual_action, manual_params, manual_log],
                outputs=[manual_log, manual_status, manual_appts, manual_params],
            )

        # ============================================================
        # TAB 3: ABOUT
        # ============================================================
        with gr.Tab("About"):
            gr.Markdown("""
### How It Works

This is an **OpenEnv-compliant evaluation environment** for testing AI agents on hospital scheduling.

**The Agentic Flow:**
1. Agent receives a task prompt (e.g., "Schedule 4 patients by urgency, verify insurance first")
2. Agent **autonomously decides** what action to take (search doctors, verify insurance, book appointment, etc.)
3. Environment processes the action and returns an observation (what happened + partial reward)
4. Agent **reads the result and reasons** about the next step
5. Loop repeats until task is complete or max steps reached
6. Grader scores the final result (0.0 to 1.0) with partial credit for sub-goals

**Key Features:**
- 7 departments, 12 doctors, 10 patients, 7 insurance providers
- Insurance validation (one plan is expired -- a trap!)
- Emergency triage (4 urgency levels)
- Waitlist management
- Doctor leave schedules
- 12 available actions
- 5 tasks: Easy -> Medium -> Hard -> Expert -> Nightmare

**The agent never sees the raw database.** It can only discover information through actions.
A smart agent verifies insurance before booking and triages by urgency.
A naive agent wastes steps and misses constraints.
            """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
