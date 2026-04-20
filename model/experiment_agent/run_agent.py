# run_agent.py - background CLI for the JamJamBeat autonomous experiment agent
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from experiment_agent import agent_graph, registry, tools
else:  # pragma: no cover - normal package import path
    from . import agent_graph, registry, tools


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model"
DEFAULT_CONFIG_PATH = MODEL_ROOT / "experiment_agent" / "agent_config.yaml"


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        candidate = cwd_candidate if cwd_candidate.exists() else PROJECT_ROOT / candidate
    return candidate.resolve()


def _default_run_id() -> str:
    return tools.now_kst().strftime("%Y%m%d_%H%M%S")


def _resolve_run_id(raw_run_id: str | None) -> str:
    return raw_run_id or tools.resolve_latest_run_id()


def _spawn_worker(run_id: str, slot: int) -> int:
    agent_dir = tools.resolve_agent_dir(run_id)
    log_path = agent_dir / "logs" / f"worker_{slot}.log"
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker", "--run-id", run_id, "--slot", str(slot)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(MODEL_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    def mutate(state: dict) -> dict:
        state = tools._set_worker_pid(state, slot, proc.pid)  # noqa: SLF001 - shared helper
        state["stop_requested"] = False
        state["status"] = "running"
        state["updated_at_kst"] = tools.iso_now()
        return state

    tools.update_state_locked(agent_dir, mutate)
    tools.append_jsonl(
        agent_dir / "decision_log.jsonl",
        {
            "event": "worker_spawned",
            "timestamp_kst": tools.iso_now(),
            "worker_pid": proc.pid,
            "worker_slot": slot,
        },
    )
    return proc.pid


def _worker_loop(run_id: str, slot: int) -> int:
    agent_dir = tools.resolve_agent_dir(run_id)
    config = tools.load_resolved_config(agent_dir)
    tools.clear_inflight_for_slot(agent_dir, slot)

    def register_worker(state: dict) -> dict:
        state = tools._set_worker_pid(state, slot, os.getpid())  # noqa: SLF001 - shared helper
        state["status"] = "running"
        state["updated_at_kst"] = tools.iso_now()
        return state

    tools.update_state_locked(agent_dir, register_worker)

    stop_path = agent_dir / "stop_requested"

    try:
        while True:
            state = tools.load_state(agent_dir)
            if str(state.get("status") or "") in tools.TERMINAL_RUN_STATUSES:
                return 0

            if str(state.get("phase_name") or "") == "mutation_search" and slot != 0:
                time.sleep(1.0)
                continue

            best_metrics = ((state.get("best") or {}).get("metrics")) or {}
            if tools.goal_reached(config["goal"], best_metrics):
                def mark_completed(payload: dict) -> dict:
                    payload["status"] = "completed"
                    payload["updated_at_kst"] = tools.iso_now()
                    return payload

                tools.update_state_locked(agent_dir, mark_completed)
                tools.append_jsonl(
                    agent_dir / "decision_log.jsonl",
                    {
                        "event": "goal_reached",
                        "timestamp_kst": tools.iso_now(),
                        "best": state.get("best"),
                        "worker_slot": slot,
                    },
                )
                return 0

            if stop_path.exists():
                def mark_stopped(payload: dict) -> dict:
                    payload["status"] = "stopped"
                    payload["stop_requested"] = True
                    payload["updated_at_kst"] = tools.iso_now()
                    return payload

                tools.update_state_locked(agent_dir, mark_stopped)
                tools.append_jsonl(
                    agent_dir / "decision_log.jsonl",
                    {
                        "event": "worker_stopped",
                        "timestamp_kst": tools.iso_now(),
                        "worker_slot": slot,
                    },
                )
                return 0

            if tools.should_wait_for_backoff(state):
                next_retry_at = state.get("next_retry_at_kst")
                now = tools.now_kst()
                wait_seconds = max(1.0, (tools.datetime.fromisoformat(next_retry_at) - now).total_seconds())
                if tools.sleep_with_stop_check(agent_dir, wait_seconds):
                    continue
                continue

            try:
                outcome = agent_graph.run_iteration.invoke({"run_dir": str(agent_dir), "worker_slot": slot})
            except RuntimeError as exc:
                if "Could not reserve a non-conflicting candidate" in str(exc):
                    tools.worker_log(
                        agent_dir,
                        "candidate reservation saturated; sleeping 5s before retry",
                        slot=slot,
                    )
                    time.sleep(5.0)
                    continue
                tools.clear_inflight_for_slot(agent_dir, slot)

                def mark_runtime_exception(payload: dict) -> dict:
                    payload["failed_attempts_consecutive"] = int(payload.get("failed_attempts_consecutive") or 0) + 1
                    payload["last_failure_at_kst"] = tools.iso_now()
                    payload["updated_at_kst"] = tools.iso_now()
                    return tools._schedule_backoff(payload, config)  # noqa: SLF001 - shared helper

                state = tools.update_state_locked(agent_dir, mark_runtime_exception)
                tools.append_jsonl(
                    agent_dir / "decision_log.jsonl",
                    {
                        "event": "worker_iteration_exception",
                        "timestamp_kst": tools.iso_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "backoff_seconds": state.get("current_backoff_seconds"),
                        "worker_slot": slot,
                    },
                )
                tools.worker_log(
                    agent_dir,
                    f"worker exception: {type(exc).__name__}: {exc} backoff={state.get('current_backoff_seconds')}s",
                    slot=slot,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive worker loop
                tools.clear_inflight_for_slot(agent_dir, slot)

                def mark_exception(payload: dict) -> dict:
                    payload["failed_attempts_consecutive"] = int(payload.get("failed_attempts_consecutive") or 0) + 1
                    payload["last_failure_at_kst"] = tools.iso_now()
                    payload["updated_at_kst"] = tools.iso_now()
                    return tools._schedule_backoff(payload, config)  # noqa: SLF001 - shared helper

                state = tools.update_state_locked(agent_dir, mark_exception)
                tools.append_jsonl(
                    agent_dir / "decision_log.jsonl",
                    {
                        "event": "worker_iteration_exception",
                        "timestamp_kst": tools.iso_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "backoff_seconds": state.get("current_backoff_seconds"),
                        "worker_slot": slot,
                    },
                )
                tools.worker_log(
                    agent_dir,
                    f"worker exception: {type(exc).__name__}: {exc} backoff={state.get('current_backoff_seconds')}s",
                    slot=slot,
                )
                continue

            state = outcome["state"]
            if state.get("status") == "completed":
                return 0
            time.sleep(0.2)
    finally:
        tools.clear_inflight_for_slot(agent_dir, slot)

        def unregister_worker(state: dict) -> dict:
            state = tools._set_worker_pid(state, slot, None)  # noqa: SLF001 - shared helper
            state["updated_at_kst"] = tools.iso_now()
            return state

        tools.update_state_locked(agent_dir, unregister_worker)


def cmd_start(args: argparse.Namespace) -> int:
    config_path = _resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw_config = tools.read_yaml(config_path)
    resolved_config = registry.resolve_agent_config(raw_config)
    run_id = args.run_id or _default_run_id()
    agent_dir = tools.resolve_agent_dir(run_id)
    if agent_dir.exists():
        raise FileExistsError(f"Agent run already exists: {agent_dir}")

    tools.setup_agent_run(agent_dir, resolved_config, source_config_path=config_path)
    parallel_workers = int(resolved_config.get("search", {}).get("parallel_workers", 1))
    pids = [_spawn_worker(run_id, slot) for slot in range(parallel_workers)]

    print(f"run_id: {run_id}")
    print(f"agent_dir: {agent_dir}")
    print(f"worker_pids: {pids}")
    print(f"log_dir: {agent_dir / 'logs'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_id = _resolve_run_id(args.run_id)
    payload = tools.build_status_payload(tools.resolve_agent_dir(run_id))
    print(tools.json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_id = _resolve_run_id(args.run_id)
    agent_dir = tools.resolve_agent_dir(run_id)
    print(tools.render_report(agent_dir, detail=args.detail))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    run_id = _resolve_run_id(args.run_id)
    agent_dir = tools.resolve_agent_dir(run_id)
    stop_path = agent_dir / "stop_requested"
    stop_path.write_text("stop_requested\n", encoding="utf-8")

    def mark_stop_requested(state: dict) -> dict:
        state["stop_requested"] = True
        state["status"] = "stop_requested"
        state["updated_at_kst"] = tools.iso_now()
        return state

    tools.update_state_locked(agent_dir, mark_stop_requested)
    tools.append_jsonl(
        agent_dir / "decision_log.jsonl",
        {
            "event": "stop_requested",
            "timestamp_kst": tools.iso_now(),
        },
    )
    print(f"stop requested: {run_id}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    run_id = _resolve_run_id(args.run_id)
    agent_dir = tools.resolve_agent_dir(run_id)
    if not agent_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {run_id}")

    state = tools.load_state(agent_dir)
    if str(state.get("status")) == "completed":
        print(f"run already completed: {run_id}")
        return 0
    worker_pids = tools._normalize_worker_pids(state)  # noqa: SLF001 - shared helper
    config = tools.load_resolved_config(agent_dir)
    parallel_workers = int(config.get("search", {}).get("parallel_workers", 1))
    alive_slots = {int(slot) for slot, pid in worker_pids.items() if tools.is_process_alive(pid)}
    if len(alive_slots) >= parallel_workers:
        print(f"workers already running: slots={sorted(alive_slots)}")
        return 0

    stop_path = agent_dir / "stop_requested"
    if stop_path.exists():
        stop_path.unlink()

    started: list[int] = []
    for slot in range(parallel_workers):
        if slot in alive_slots:
            continue
        started.append(_spawn_worker(run_id, slot))
    print(f"resumed run_id={run_id} worker_pids={started}")
    print(f"log_dir: {agent_dir / 'logs'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the JamJamBeat autonomous experiment agent until target test metrics are met.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Create a new autonomous experiment run.")
    start.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    start.add_argument("--run-id", type=str, default="")
    start.set_defaults(func=cmd_start)

    status = subparsers.add_parser("status", help="Show current summary status.")
    status.add_argument("--run-id", type=str, default="")
    status.set_defaults(func=cmd_status)

    report = subparsers.add_parser("report", help="Render a saved report.")
    report.add_argument("--run-id", type=str, default="")
    report.add_argument("--detail", choices=("summary", "full"), default="summary")
    report.set_defaults(func=cmd_report)

    stop = subparsers.add_parser("stop", help="Request a graceful stop.")
    stop.add_argument("--run-id", type=str, default="")
    stop.set_defaults(func=cmd_stop)

    resume = subparsers.add_parser("resume", help="Resume an existing stopped run.")
    resume.add_argument("--run-id", type=str, default="")
    resume.set_defaults(func=cmd_resume)

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-id", type=str, required=True)
    worker.add_argument("--slot", type=int, required=True)
    worker.set_defaults(func=lambda args: _worker_loop(args.run_id, args.slot))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
