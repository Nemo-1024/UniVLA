"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import jsonlines
import numpy as np
import torch
import wandb

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Define Tracker Interface ===
class Tracker(Protocol):
    def write_hyperparameters(self) -> None: ...

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None: ...

    def finalize(self) -> None: ...


# === Individual Tracker Definitions ===
class JSONLinesTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        with jsonlines.open(self.run_dir / "run-metrics.jsonl", mode="w", sort_keys=True) as js_tracker:
            js_tracker.write({"run_id": self.run_id, "hparams": self.hparams})

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        with jsonlines.open(self.run_dir / f"{self.run_id}.jsonl", mode="a", sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return


class WeightsBiasesTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "prismatic",
        entity: Optional[str] = None,
        group: str = "align",
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Get W&B-Specific Initialization Parameters
        self.project, self.entity, self.group, self.wandb_dir = project, entity, group, self.run_dir

        # Call W&B.init()
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if overwatch.is_rank_zero():
            try:
                wandb.finish()
            except Exception:
                pass
        # 移除长时间 sleep，避免阻塞
        return


class AccelerateTracker:
    def __init__(self, accelerator, run_id: str, hparams: Dict[str, Any], stage: str) -> None:
        self.accelerator = accelerator
        self.run_id = run_id
        self.hparams = hparams
        self.stage = stage

    def write_hyperparameters(self) -> None:
        # 由外部 accelerator.init_trackers 写入，这里无需重复
        return

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        # 直接通过 accelerator.log 推送
        safe_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        self.accelerator.log(safe_metrics, step=global_step)

    def finalize(self) -> None:
        # 由加速器结束时统一处理
        return


# === Core Metrics Container :: Initializes Trackers => Compiles/Pushes Metrics ===


class Metrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        stage: str,
        wandb_project: str = "prismatic",
        wandb_entity: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        window_size: int = 128,
        accelerator: Optional[Any] = None,
    ) -> None:
        self.run_id, self.run_dir, self.hparams, self.stage = run_id, run_dir, hparams, stage

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity, group=self.stage
                )
            elif tracker_type == "accelerate" and accelerator is not None:
                tracker = AccelerateTracker(accelerator, run_id, hparams, stage)
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step, self.start_time, self.step_start_time = 0, time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "l1_loss": deque(maxlen=window_size),
            "action_accuracy": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

        # Per-dataset rolling buffers for action metrics
        self.dataset_state: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {"l1_loss": deque(maxlen=window_size), "action_accuracy": deque(maxlen=window_size)}
        )

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        safe_metrics: Dict[str, Union[int, float]] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                safe_metrics[k] = v
            elif hasattr(v, "item"):
                try:
                    safe_metrics[k] = float(v.item())
                except Exception:
                    continue
            else:
                try:
                    safe_metrics[k] = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
        for tracker in self.trackers:
            tracker.write(global_step, safe_metrics)

    def get_status(self, loss: Optional[float] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Global Step] {self.global_step:06d} =>> LR :: {lr:.6f} -- Loss :: {loss:.4f}"

    def commit(
        self, *, global_step: Optional[int] = None, lr: Optional[float] = None, update_step_time: bool = False, **kwargs
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach() if hasattr(value, "detach") else torch.tensor(float(value))
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            elif key in {"l1_loss", "action_accuracy"}:
                val_t = value.detach() if hasattr(value, "detach") else torch.tensor(float(value))
                self.state[key].append(val_t)
            else:
                val_t = value.detach() if hasattr(value, "detach") else torch.tensor(float(value))
                self.state.setdefault(key, deque(maxlen=128)).append(val_t)

    def commit_for_dataset(self, dataset_name: str, **kwargs) -> None:
        if not overwatch.is_rank_zero():
            return
        ds_buf = self.dataset_state[dataset_name]
        for key, value in kwargs.items():
            if key in {"l1_loss", "action_accuracy"}:
                val = value.detach().item() if hasattr(value, "detach") else float(value)
                ds_buf[key].append(val)

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item() if len(self.state["loss_raw"]) > 0 else 0.0
        loss = torch.stack(list(self.state["loss"])).mean().item() if len(self.state["loss"]) > 0 else 0.0
        l1_loss = (
            torch.stack(list(self.state["l1_loss"])).mean().item() if len(self.state["l1_loss"]) > 0 else 0.0
        )
        action_accuracy = (
            torch.stack(list(self.state["action_accuracy"])).mean().item()
            if len(self.state["action_accuracy"]) > 0
            else 0.0
        )
        step_time, lr = np.mean(list(self.state["step_time"])), self.state["lr"][-1]
        status = self.get_status(loss)

        # Fire to Trackers
        prefix = self.stage.capitalize()
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": int(self.global_step),
                f"{prefix}/Loss": float(loss),
                f"{prefix}/L1 Loss": float(l1_loss),
                f"{prefix}/Action Token Accuracy": float(action_accuracy),
                f"{prefix}/Loss (Raw)": float(loss_raw),
                f"{prefix}/Learning Rate": float(lr),
                f"{prefix}/Step Time": float(step_time),
            },
        )

        # Per-dataset metrics
        if len(self.dataset_state) > 0:
            ds_metrics: Dict[str, float] = {}
            for ds, buf in self.dataset_state.items():
                if len(buf["l1_loss"]) > 0:
                    ds_metrics[f"{ds}/L1 Loss"] = float(np.mean(list(buf["l1_loss"])))
                if len(buf["action_accuracy"]) > 0:
                    ds_metrics[f"{ds}/Action Token Accuracy"] = float(np.mean(list(buf["action_accuracy"])))
            if len(ds_metrics) > 0:
                self.log(self.global_step, metrics=ds_metrics) 
        return status

    def finalize(self) -> None:
        for tracker in self.trackers:
            tracker.finalize()


class VLAMetrics:
    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        wandb_project: str = "openvla",
        wandb_entity: Optional[str] = "stanford-voltron",
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: Optional[int] = None,
        resume_epoch: Optional[int] = None,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == "jsonl":
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == "wandb":
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity, group="vla-train"
                )
            else:
                raise ValueError(f"Tracker with type `{tracker_type} is not supported!")

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch
        self.start_time, self.step_start_time = time.time(), time.time()
        self.state = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "l1_loss": deque(maxlen=window_size),
            "action_accuracy": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

        # Created metrics buffers for individual tracked datasets
        self.dataset_trackers = defaultdict(lambda: VLAMetrics((), "", Path("."), {}))

    def log(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        safe_metrics: Dict[str, Union[int, float]] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                safe_metrics[k] = v
            elif hasattr(v, "item"):
                try:
                    safe_metrics[k] = float(v.item())
                except Exception:
                    continue
            else:
                try:
                    safe_metrics[k] = float(v)  # type: ignore[arg-type]
                except Exception:
                    continue
        for tracker in self.trackers:
            tracker.write(global_step, safe_metrics)

    def get_status(self, loss: Optional[float] = None) -> str:
        lr = self.state["lr"][-1] if len(self.state["lr"]) > 0 else 0
        if loss is None:
            return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} - Loss :: {loss:.4f}"

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        update_step_time: bool = False,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss_raw"].append(loss_val)
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value.detach())

    def commit_for_dataset(self, dataset_name: str, **kwargs) -> None:
        self.dataset_trackers[dataset_name].commit(**kwargs)

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state["loss_raw"])).mean().item()
        loss = torch.stack(list(self.state["loss"])).mean().item()
        l1_loss = torch.stack(list(self.state["l1_loss"])).mean().item()
        action_accuracy = torch.stack(list(self.state["action_accuracy"])).mean().item()
        step_time, lr = np.mean(list(self.state["step_time"])), self.state["lr"][-1]
        status = self.get_status(loss)

        # Get metrics per dataset
        dataset_metrics: Dict[str, float] = {}
        for ds, tracker in self.dataset_trackers.items():
            dataset_metrics.update(
                {
                    f"{ds}/L1 Loss": float(
                        torch.stack(list(tracker.state["l1_loss"])) .mean().item()
                        if len(tracker.state.get("l1_loss", [])) > 0
                        else 0.0
                    ),
                    f"{ds}/Action Token Accuracy": float(
                        torch.stack(list(tracker.state["action_accuracy"])) .mean().item()
                        if len(tracker.state.get("action_accuracy", [])) > 0
                        else 0.0
                    ),
                }
            )

        # Fire to Trackers
        prefix = "VLA Train"
        self.log(
            self.global_step,
            metrics={
                f"{prefix}/Step": int(self.global_step),
                f"{prefix}/Epoch": int(self.epoch),
                f"{prefix}/Loss": float(loss),
                f"{prefix}/L1 Loss": float(l1_loss),
                f"{prefix}/Action Token Accuracy": float(action_accuracy),
                f"{prefix}/Loss (Raw)": float(loss_raw),
                f"{prefix}/Learning Rate": float(lr),
                f"{prefix}/Step Time": float(step_time),
                **dataset_metrics,
            },
        )
        return status

    def finalize(self) -> None:
        for tracker in self.trackers:
            tracker.finalize()
