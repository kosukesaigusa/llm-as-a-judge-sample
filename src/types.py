from typing import TypedDict, List, Literal

class PromptItem(TypedDict):
    """プロンプトの各ターンを表す辞書型。"""
    role: Literal["user", "assistant"]
    content: str

class RubricItem(TypedDict):
    """ルーブリック（評価基準）の各項目を表す辞書型。"""
    criterion: str
    points: int

class PromptDatasetItem(TypedDict):
    """LLM応答作成前のデータ項目（プロンプトのみ）。"""
    prompt_id: str
    prompts: List[PromptItem]
    rubrics: List[RubricItem]

class EvaluationDatasetItem(PromptDatasetItem):
    """評価対象データ項目（LLM応答付き）。"""
    llm_response_text: str

class RatingResult(TypedDict):
    """主観評価・自由記述評価の結果（スコアと説明）。"""
    explanation: str
    rating: int

class RubricResult(TypedDict):
    """ルーブリック評価の結果（適合判定と説明）。"""
    explanation: str
    criteriaMet: bool
