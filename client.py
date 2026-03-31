"""Hospital Scheduler Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import HospitalAction, HospitalObservation, HospitalState


class HospitalSchedulerEnv(
    EnvClient[HospitalAction, HospitalObservation, HospitalState]
):
    """Client for the Hospital Scheduler Environment.

    Example:
        >>> with HospitalSchedulerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="task_easy")
        ...     print(result.observation.message)
        ...
        ...     action = HospitalAction(action_type="get_patient_info", parameters={"patient_id": "P001"})
        ...     result = client.step(action)
        ...     print(result.observation.message)
    """

    def _step_payload(self, action: HospitalAction) -> Dict:
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HospitalObservation]:
        obs_data = payload.get("observation", {})
        observation = HospitalObservation(
            status=obs_data.get("status", "success"),
            message=obs_data.get("message", ""),
            data=obs_data.get("data", {}),
            available_actions=obs_data.get("available_actions", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> HospitalState:
        return HospitalState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_prompt=payload.get("task_prompt", ""),
            max_steps=payload.get("max_steps", 0),
            done_flag=payload.get("done_flag", False),
            current_reward=payload.get("current_reward", 0.0),
        )
