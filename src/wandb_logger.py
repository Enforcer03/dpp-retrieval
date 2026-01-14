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


def wandb_log_artifact(run: Any, file_path: str, artifact_name: str, artifact_type: str = "output") -> None:
    """Log file artifact to wandb."""
    if run is None:
        return
    try:
        import wandb  # type: ignore
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            log.warning("Artifact file not found: %s", file_path)
            return

        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        log.debug("Logged artifact: %s", artifact_name)
    except Exception as e:
        log.warning("wandb artifact logging failed for %s: %s", artifact_name, e)


def wandb_log_image(
    run: Any,
    image_key: str,
    image_or_path: Any,
    caption: str = "",
) -> None:
    """Log image directly to wandb dashboard (not as artifact).

    Args:
        run: wandb run object
        image_key: Key name for the image in wandb
        image_or_path: Either file path, PIL Image, or matplotlib figure
        caption: Optional caption for the image
    """
    if run is None:
        return
    try:
        import wandb  # type: ignore

        wandb.log({image_key: wandb.Image(image_or_path, caption=caption)})
        log.debug("Logged image: %s", image_key)
    except Exception as e:
        log.warning("Failed to log image to wandb (%s): %s", image_key, e)


def wandb_set_summary(run: Any, summary_dict: dict) -> None:
    """Set summary metrics for wandb run (appears in Overview tab).

    Args:
        run: wandb run object
        summary_dict: Dict of key-value pairs to set as summary
    """
    if run is None:
        return
    try:
        import wandb  # type: ignore

        for key, value in summary_dict.items():
            run.summary[key] = value
        log.debug("Set wandb summary metrics: %s", list(summary_dict.keys()))
    except Exception as e:
        log.warning("Failed to set wandb summary: %s", e)


def wandb_finish(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
