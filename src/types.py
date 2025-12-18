from typing import Literal, TypedDict

from pydantic import BaseModel, computed_field


class PromptItem(TypedDict):
    """プロンプトの各ターンを表す辞書型。"""

    role: Literal["user", "assistant"]
    content: str


class RubricItem(TypedDict):
    """ルーブリック（評価基準）の各項目を表す辞書型。"""

    criterion: str
    points: int


class GenerationDatasetItem(TypedDict):
    """LLM応答作成前のデータ項目（プロンプトとシステム指示）。"""

    prompts: list[PromptItem]
    generator_system_instructions: list[str]


class PromptDatasetItem(TypedDict):
    """LLM応答作成前のデータ項目（プロンプトのみ）。"""

    prompts: list[PromptItem]
    rubrics: list[RubricItem]


class EvaluationDatasetItem(PromptDatasetItem):
    """評価対象データ項目（LLM応答付き）。"""

    llm_response_text: str


class RatingResult(TypedDict):
    """主観評価・自由記述評価の結果（点数と説明）。"""

    explanation: str
    rating: int


class RubricResult(TypedDict):
    """ルーブリック評価の結果（適合判定と説明）。"""

    explanation: str
    criteria_met: bool


class EvaluationResultByRubric(BaseModel):
    """ルーブリックごとの評価結果。"""

    rubric: RubricItem
    explanation: str
    criteria_met: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def signed_score(self) -> int:
        """当該ルーブリックに対する点数（criteria_met に応じた加点または減点）を取得する。"""
        return self.rubric["points"] if self.criteria_met else 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_criteria_passed(self) -> bool:
        """
        当該ルーブリックの基準に合格しているかどうか。

        ルーブリックのポイントが
        - 正の数の場合は、criteria_met が true であること
        - 負の数の場合は、criteria_met が false であること
        を意味する。
        """
        if self.rubric["points"] == 0:
            raise ValueError("Rubric の points は正の数または負の数である必要があります。")
        return self.criteria_met if self.rubric["points"] > 0 else not self.criteria_met


class EvaluationOutput(BaseModel):
    """評価結果。"""

    prompt_id: str
    prompts: list[PromptItem]
    llm_response_text: str
    result_by_rubrics: list[EvaluationResultByRubric]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_score(self) -> int:
        """総合点数。"""
        return sum(result.signed_score for result in self.result_by_rubrics)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def theoretical_score(self) -> int:
        """理論上の最大点数（ルーブリックの得点が正のものの合計）。"""
        return sum(result.rubric["points"] for result in self.result_by_rubrics if result.rubric["points"] > 0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def score_rate(self) -> float:
        """理論上の最大点数に対する総合点数の割合。"""
        if self.theoretical_score > 0:
            return self.total_score / self.theoretical_score
        return 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def criteria_pass_rate(self) -> float:
        """各ルーブリックの基準に合格している割合。"""
        if len(self.result_by_rubrics) == 0:
            return 0.0
        return sum(1 for r in self.result_by_rubrics if r.is_criteria_passed) / len(self.result_by_rubrics)
