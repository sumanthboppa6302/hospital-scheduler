"""OpenEnv Environment wrapper for the hospital scheduler."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import HospitalEnv
from graders import grade
from models import HospitalAction, HospitalObservation, HospitalState, ActionType

TASK_IDS = ["task_easy", "task_medium", "task_hard", "task_expert", "task_nightmare"]


class HospitalEnvironment(Environment[HospitalAction, HospitalObservation, HospitalState]):
    """Hospital appointment scheduling environment for OpenEnv."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._env = HospitalEnv()
        self._state = HospitalState(episode_id=str(uuid4()), step_count=0)
        self._current_task_id: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HospitalObservation:
        task_id = kwargs.get("task_id", "task_easy")
        if task_id not in TASK_IDS:
            return HospitalObservation(
                status="error",
                message=f"Unknown task: {task_id}. Available: {TASK_IDS}",
                done=True,
                reward=0.0,
            )

        self._current_task_id = task_id
        obs = self._env.reset(task_id)
        self._state = HospitalState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            task_prompt=obs.message,
            max_steps=obs.max_steps,
            done_flag=False,
            current_reward=0.0,
        )

        return HospitalObservation(
            status=obs.status,
            message=obs.message,
            data=obs.data,
            available_actions=obs.available_actions,
            done=False,
            reward=0.0,
            step_number=0,
            max_steps=obs.max_steps,
        )

    def step(
        self,
        action: HospitalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HospitalObservation:
        try:
            action_type = ActionType(action.action_type)
        except ValueError:
            valid = [a.value for a in ActionType]
            return HospitalObservation(
                status="error",
                message=f"Unknown action_type: {action.action_type}. Valid: {valid}",
                done=self._env.done,
                reward=self._env.current_reward,
                step_number=self._env.step_number,
                max_steps=self._env.task_config.max_steps if self._env.task_config else 0,
            )

        from env import Action as InternalAction
        internal_action = InternalAction(action_type=action_type, parameters=action.parameters)
        obs = self._env.step(internal_action)

        self._state.step_count = self._env.step_number
        self._state.done_flag = obs.done
        self._state.current_reward = obs.reward

        return HospitalObservation(
            status=obs.status,
            message=obs.message,
            data=obs.data,
            available_actions=obs.available_actions,
            done=obs.done,
            reward=obs.reward,
            step_number=obs.step_number,
            max_steps=obs.max_steps,
        )

    @property
    def state(self) -> HospitalState:
        s = self._env.state()
        return HospitalState(
            episode_id=self._state.episode_id,
            step_count=self._env.step_number,
            task_id=s.task_id,
            task_prompt=s.task_prompt,
            max_steps=s.max_steps,
            done_flag=s.done,
            current_reward=s.current_reward,
            appointments=[a.model_dump() for a in s.appointments],
            waitlist=[w.model_dump() for w in s.waitlist],
            action_history=[a.model_dump() for a in s.action_history],
        )

    def get_grade(self) -> float:
        return grade(self._env)

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="HospitalScheduler",
            description="Hospital appointment scheduling with insurance, triage, and waitlist management",
            version="2.0.0",
        )
