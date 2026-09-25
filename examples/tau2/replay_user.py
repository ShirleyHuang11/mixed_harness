"""A tau2 user that replays the recorded user turns of an earlier simulation, for offline plumbing runs.

The replayed user says what the recorded user said, in order, whatever the agent replies, and stops when the
recording is exhausted. Rewards from such a run mean nothing (the conversation is not coherent); the counters and
the step table do, which is what a plumbing run is for. Register it and select it with `user='replay_user'`:

    ReplayUser.load('runs/tau2/<stamp>/llm_only/results.json')
    registry.register_user(ReplayUser, 'replay_user')
"""
import json
from pathlib import Path

from tau2.data_model.message import UserMessage
from tau2.user.user_simulator_base import STOP, HalfDuplexUser, UserState


class ReplayUser(HalfDuplexUser[UserState]):
    recordings = {}        # task id -> [user text, ...]
    scenario_to_task = {}  # str(task.user_scenario) -> task id; tau2 builds the user with instructions=str(task.user_scenario)

    def __init__(self, instructions=None, tools=None, llm=None, llm_args=None, **kwargs):
        super().__init__(instructions=instructions, tools=tools)
        self.llm = llm or 'replay'
        self.task_id = self.scenario_to_task.get(instructions)
        self._queue = None

    @classmethod
    def bind(cls, tasks):
        """Map each tau2 Task's scenario text (what build_user passes as instructions) to its id."""
        cls.scenario_to_task = {str(t.user_scenario): str(t.id) for t in tasks}

    @classmethod
    def load(cls, path):
        results = json.loads(Path(path).read_text())
        for sim in results.get('simulations') or []:
            texts = [m.get('content') or '' for m in (sim.get('messages') or []) if m.get('role') == 'user']
            cls.recordings[str(sim.get('task_id'))] = texts
        return cls.recordings

    @classmethod
    def is_stop(cls, message):
        return message.content is not None and STOP in message.content

    def set_seed(self, seed):
        pass

    def get_init_state(self, message_history=None):
        return UserState(messages=[], system_messages=[])

    def generate_next_message(self, message, state):
        if self._queue is None:
            texts = self.recordings.get(self.task_id)
            if texts is None:
                raise ValueError(f'no recording for task {self.task_id!r}; call ReplayUser.bind(tasks) and load a results.json that contains it')
            self._queue = list(texts)
        content = self._queue.pop(0) if self._queue else STOP
        return UserMessage(role='user', content=content, cost=0.0), state
