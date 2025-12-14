import json
import os
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo
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


def save_evaluation_dataset(
    data: List[EvaluationDatasetItem],
    file_path: Optional[str] = None
) -> str:
    """
    評価対象データを保存する。
    
    Args:
        data: 保存する評価対象データ
        file_path: 保存先のファイルパス。None の場合はタイムスタンプ付きのファイル名を生成する。
    
    Returns:
        保存したファイルのパス
    """
    if file_path is None:
        # 日本時間でタイムスタンプを生成する。
        jst = ZoneInfo("Asia/Tokyo")
        timestamp = datetime.now(jst).strftime("%Y-%m-%d-%H-%M-%S")
        file_path = os.path.join(DATA_DIR, f"{timestamp}.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return file_path
