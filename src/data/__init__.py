import json
import os
from typing import List
from ..types import EvaluationDatasetItem, PromptDatasetItem

DATA_DIR = os.path.dirname(__file__)
EVALUATION_DATASET_PATH = os.path.join(DATA_DIR, "evaluation_dataset.json")
PROMPT_DATASET_PATH = os.path.join(DATA_DIR, "prompt_dataset.json")


def get_evaluation_dataset() -> List[EvaluationDatasetItem]:
    """評価対象データ（LLM応答付き）を取得する。"""
    with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_prompt_dataset() -> List[PromptDatasetItem]:
    """プロンプトデータ（LLM応答なし）を取得する。"""
    with open(PROMPT_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_evaluation_dataset(data: List[EvaluationDatasetItem]) -> None:
    """評価対象データを保存する。"""
    with open(EVALUATION_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
