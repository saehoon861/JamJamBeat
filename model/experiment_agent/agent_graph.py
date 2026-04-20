# agent_graph.py - LangGraph 1.0 functional workflow for one autonomous experiment iteration
from __future__ import annotations

from pathlib import Path

from langgraph.func import entrypoint, task

from . import tools


@task
def propose_candidate_task(run_dir: str, worker_slot: int | None = None) -> dict:
    return tools.propose_candidate(Path(run_dir), worker_slot=worker_slot)


@task
def execute_candidate_task(run_dir: str, candidate: dict) -> dict:
    return tools.execute_candidate(Path(run_dir), candidate)


@task
def update_after_trial_task(run_dir: str, candidate: dict, result: dict) -> dict:
    return tools.update_after_trial(Path(run_dir), candidate, result)


@entrypoint()
def run_iteration(payload: dict) -> dict:
    run_dir = str(payload["run_dir"])
    worker_slot = payload.get("worker_slot")
    candidate = propose_candidate_task(run_dir, worker_slot=worker_slot).result()
    result = execute_candidate_task(run_dir, candidate).result()
    state = update_after_trial_task(run_dir, candidate, result).result()
    return {
        "candidate": candidate,
        "result": result,
        "state": state,
    }
