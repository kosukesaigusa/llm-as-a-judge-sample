from typing import List, Optional
from datetime import datetime
from ..models import generate
from .prompt import (
    SUBJECTIVE_EVALUATION_PROMPT_TEMPLATE,
    GENERAL_EVALUATION_PROMPT_TEMPLATE,
    RUBRIC_EVALUATION_PROMPT_TEMPLATE,
)
from ..types import (
    RubricItem,
    RatingResult,
    EvaluationDatasetItem,
    EvaluationResultByRubric,
    EvaluationOutput,
)

# 評価の出力スキーマ。
RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "rating": {"type": "integer"},
    },
    "required": ["explanation", "rating"],
}

# ルーブリック評価の出力スキーマ。
RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "criteria_met": {"type": "boolean"},
    },
    "required": ["explanation", "criteria_met"],
}


def _generate_with_retry(
    model_name: str,
    prompt: str,
    schema: dict,
    required_keys: List[str],
    max_retries: int = 5,
    show_available_keys: bool = False
) -> Optional[dict]:
    """
    指定されたスキーマに従った JSON を生成し、必要なキーが存在するまでリトライする。
    
    Args:
        model_name: モデル名
        prompt: プロンプト
        schema: JSON スキーマ
        required_keys: 結果に含まれる必要があるキーのリスト
        max_retries: 最大リトライ回数（デフォルト: 5）
        show_available_keys: キーが不足している場合に利用可能なキーを表示するか（デフォルト: False）
    
    Returns:
        期待通りの JSON が取得できた場合は辞書、それ以外は None
    """
    result = None
    for attempt in range(1, max_retries + 1):
        result = generate(model_name, prompt=prompt, schema=schema, temperature=0)
        
        # 期待通りの JSON かどうかを確認する。
        if result is not None and isinstance(result, dict):
            if all(key in result for key in required_keys):
                # 期待通りの JSON が取得できた。
                break
            else:
                # 必要なキーが存在しない。
                if attempt < max_retries:
                    print(f"Warning: Attempt {attempt}/{max_retries} - Missing required keys in result. Retrying...")
                    if show_available_keys:
                        print(f"Available keys: {list(result.keys())}")
                result = None
        else:
            # None または辞書型でない。
            if attempt < max_retries:
                print(f"Warning: Attempt {attempt}/{max_retries} - Failed to get valid result. Retrying...")
            result = None
    
    # 最終的な結果が期待通りかどうかを確認する。
    if result is not None and isinstance(result, dict) and all(key in result for key in required_keys):
        return result
    return None


def run_subjective_evaluation(
    conversation: str,
    model_name: str
) -> Optional[RatingResult]:
    """主観評価を実行する。"""
    prompt = SUBJECTIVE_EVALUATION_PROMPT_TEMPLATE.replace("<<conversation>>", conversation)
    result = _generate_with_retry(
        model_name=model_name,
        prompt=prompt,
        schema=RATING_SCHEMA,
        required_keys=['explanation', 'rating']
    )
    return result if isinstance(result, dict) else None


def run_general_evaluation(
    conversation: str,
    model_name: str
) -> Optional[RatingResult]:
    """自由記述評価を実行する。"""
    prompt = GENERAL_EVALUATION_PROMPT_TEMPLATE.replace("<<conversation>>", conversation)
    
    result = _generate_with_retry(
        model_name=model_name,
        prompt=prompt,
        schema=RATING_SCHEMA,
        required_keys=['explanation', 'rating']
    )
    return result if isinstance(result, dict) else None


def run_rubric_evaluation(
    data: EvaluationDatasetItem,
    model_name: str,
    prompt_id: Optional[str] = None
) -> Optional[EvaluationOutput]:
    """
    ルーブリック評価を実行して EvaluationOutput を返す。
    
    Args:
        data: 評価対象データ項目
        model_name: 評価に使用するモデル名
        prompt_id: プロンプト ID（指定されない場合は現在時刻から生成）
    
    Returns:
        評価結果、または None（評価に失敗した場合）
    """
    if prompt_id is None:
        prompt_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # 会話履歴の構築
    conversation = ""
    for p in data['prompts']:
        conversation += f"{p['role']}: {p['content']}\n"
    conversation += f"assistant: {data['llm_response_text']}"
    
    # 各ルーブリックに対して評価を実行する。
    result_by_rubrics: List[EvaluationResultByRubric] = []
    for rubric_item in data['rubrics']:
        # criterionWithPoints の形式: "[points] criterion"
        criterion_text = f"[{rubric_item['points']}] {rubric_item['criterion']}"
        
        prompt = RUBRIC_EVALUATION_PROMPT_TEMPLATE.replace("<<conversation>>", conversation) \
            .replace("<<rubric_item>>", criterion_text)
        
        result = _generate_with_retry(
            model_name=model_name,
            prompt=prompt,
            schema=RUBRIC_SCHEMA,
            required_keys=['explanation', 'criteria_met'],
            show_available_keys=True
        )
        
        # リトライ後も期待通りの JSON が取得できなかった場合。
        if result is None:
            print(f"Error: Failed to get valid result after specified number of attempts for criterion: {rubric_item['criterion']}")
            return None
        
        result_by_rubrics.append(
            EvaluationResultByRubric(
                rubric=rubric_item,
                explanation=result['explanation'],
                criteria_met=result['criteria_met']
            )
        )
    
    return EvaluationOutput(
        prompt_id=prompt_id,
        prompts=data['prompts'],
        llm_response_text=data['llm_response_text'],
        result_by_rubrics=result_by_rubrics
    )

