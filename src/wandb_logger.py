from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


def wandb_init(enabled: bool, project: str, entity: str | None, tags: list[str], name: str | None, config: dict) -> Any:
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        log.warning("wandb unavailable: %s", e)
        return None

    proj = project or os.getenv("WANDB_PROJECT", "icb-sum")
    ent = entity or os.getenv("WANDB_ENTITY")
    try:
        return wandb.init(project=proj, entity=ent, name=name, tags=tags, config=config, reinit=True)
    except Exception as e:
        log.warning("wandb init failed: %s", e)
        return None


def wandb_log(run: Any, metrics: dict) -> None:
    if run is None:
        return
    try:
        import wandb  # type: ignore

        wandb.log(metrics)
    except Exception:
        pass


def wandb_finish(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
