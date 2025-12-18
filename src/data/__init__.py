import json
import os
from typing import cast

from ..types import EvaluationDatasetItem, GenerationDatasetItem, RubricItem

DATA_DIR = os.path.dirname(__file__)
EVALUATION_DATASET_PATH = os.path.join(DATA_DIR, "evaluation_dataset.json")
GENERATION_DATASET_PATH = os.path.join(DATA_DIR, "generation_dataset.json")
RUBRICS_PATH = os.path.join(DATA_DIR, "rubrics.json")


def get_evaluation_dataset() -> list[EvaluationDatasetItem]:
    """評価対象データ（LLM応答付き）を取得する。"""
    with open(EVALUATION_DATASET_PATH, encoding="utf-8") as f:
        return cast(list[EvaluationDatasetItem], json.load(f))


def get_generation_dataset() -> list[GenerationDatasetItem]:
    """生成用データ（プロンプトとシステム指示）を取得する。"""
    with open(GENERATION_DATASET_PATH, encoding="utf-8") as f:
        return cast(list[GenerationDatasetItem], json.load(f))


def get_rubrics() -> list[RubricItem]:
    """共通のルーブリックを取得する。"""
    with open(RUBRICS_PATH, encoding="utf-8") as f:
        return cast(list[RubricItem], json.load(f))


def save_evaluation_dataset(data: list[EvaluationDatasetItem]) -> None:
    """
    評価対象データを保存する。

    Args:
        data: 保存する評価対象データ
    """
    with open(EVALUATION_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
