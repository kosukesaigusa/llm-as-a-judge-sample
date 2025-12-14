import json
from typing import List, Optional, Any, Dict
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from .prompt import (
    SUBJECTIVE_PROMPT_TEMPLATE,
    FREE_FORM_PROMPT_TEMPLATE,
    RUBRIC_PROMPT_TEMPLATE,
)
from ..types import RubricItem, RatingResult, RubricResult

# 評価の出力スキーマ。
RATING_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "explanation": {"type": "STRING"},
        "rating": {"type": "INTEGER"},
    },
    "required": ["explanation", "rating"],
}

# ルーブリック評価の出力スキーマ。
RUBRIC_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "explanation": {"type": "STRING"},
        "criteriaMet": {"type": "BOOLEAN"},
    },
    "required": ["explanation", "criteriaMet"],
}


def init_vertexai(project_id: str, location: str) -> None:
    """Vertex AI を初期化する。"""
    vertexai.init(project=project_id, location=location)


def get_model(model_name: str = "gemini-2.5-flash") -> GenerativeModel:
    """モデルインスタンスを取得する。"""
    return GenerativeModel(model_name)


def generate_json(model: GenerativeModel, prompt: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """JSON スキーマに基づいて LLM から JSON を生成する。"""
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=schema
    )
    response = model.generate_content(prompt, generation_config=generation_config)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {response.text}")
        return None


def run_subjective_evaluation(conversation: str, model_name: str = "gemini-2.5-flash") -> Optional[RatingResult]:
    """主観評価を実行する。"""
    prompt = SUBJECTIVE_PROMPT_TEMPLATE.replace("<<conversation>>", conversation)
    model = get_model(model_name)
    result = generate_json(model, prompt, RATING_SCHEMA)
    return result  # type: ignore


def run_free_form_evaluation(conversation: str, rubrics: List[RubricItem], model_name: str = "gemini-2.5-flash") -> Optional[RatingResult]:
    """自由記述評価を実行する。"""
    positive_rubrics = [r for r in rubrics if r['points'] > 0]
    negative_rubrics = [r for r in rubrics if r['points'] < 0]
    
    if positive_rubrics:
        pos_section = "\n".join([f"- {r['criterion']}" for r in positive_rubrics])
    else:
        pos_section = "- There are no explicit positive criteria. Focus only on the overall usefulness and quality of the response."
        
    if negative_rubrics:
        neg_section = "\n".join([f"- {r['criterion']}" for r in negative_rubrics])
    else:
        neg_section = "- There are no explicit negative criteria. Only check that there are no obvious errors or harmful issues."

    prompt = FREE_FORM_PROMPT_TEMPLATE.replace("<<conversation>>", conversation) \
        .replace("<<positive_criteria>>", pos_section) \
        .replace("<<negative_criteria>>", neg_section)
        
    model = get_model(model_name)
    result = generate_json(model, prompt, RATING_SCHEMA)
    return result  # type: ignore


def run_rubric_evaluation(conversation: str, rubric_item: RubricItem, model_name: str = "gemini-2.5-flash") -> Optional[RubricResult]:
    """ルーブリック評価を実行する。"""
    # criterionWithPoints の形式: "[points] criterion"
    criterion_text = f"[{rubric_item['points']}] {rubric_item['criterion']}"
    
    prompt = RUBRIC_PROMPT_TEMPLATE.replace("<<conversation>>", conversation) \
        .replace("<<rubric_item>>", criterion_text)
        
    model = get_model(model_name)
    result = generate_json(model, prompt, RUBRIC_SCHEMA)
    return result  # type: ignore

