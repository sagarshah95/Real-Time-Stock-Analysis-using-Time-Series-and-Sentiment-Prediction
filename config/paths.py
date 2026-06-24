"""Resolve project data paths with legacy folder fallbacks."""

from __future__ import annotations

from pathlib import Path

from config.settings import PROJECT_ROOT


def first_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def transcripts_dir(configured: str | Path | None = None) -> Path:
    preferred = Path(configured) if configured else PROJECT_ROOT / "data" / "transcripts"
    return first_existing_path(
        preferred,
        PROJECT_ROOT / "data" / "transcripts",
        PROJECT_ROOT / "inference-data",
    )


def sp500_csv_path(configured: str | Path | None = None) -> Path:
    preferred = Path(configured) if configured else PROJECT_ROOT / "data" / "datasets" / "SP500.csv"
    return first_existing_path(
        preferred,
        PROJECT_ROOT / "data" / "datasets" / "SP500.csv",
        PROJECT_ROOT / "Datasets" / "SP500.csv",
    )


def architecture_image_path() -> Path | None:
    for path in (
        PROJECT_ROOT / "assets" / "images" / "architecture-aws-fast.jpg",
        PROJECT_ROOT / "assets" / "images" / "Architecture Final AWS_FAST.jpg",
        PROJECT_ROOT / "Images" / "Architecture Final AWS_FAST.jpg",
    ):
        if path.is_file():
            return path
    return None


def docker_compose_kafka_path() -> Path:
    return first_existing_path(
        PROJECT_ROOT / "infra" / "docker-compose.kafka.yml",
        PROJECT_ROOT / "docker-compose.kafka.yml",
    )
