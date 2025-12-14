from typing import List, cast
from vertexai.generative_models import Content, Part, GenerativeModel
from src.data import get_prompt_dataset, save_evaluation_dataset
from src.evaluator.evaluator import init_vertexai
from src.types import EvaluationDatasetItem, PromptItem
from .prompt import SYSTEM_INSTRUCTION

def _build_conversation_contents(prompts: List[PromptItem]) -> List[Content]:
    """
    プロンプトリストから会話履歴の Content リストを構築する。
    
    Returns:
        contents: Content リスト
    """
    contents: List[Content] = []
    
    for prompt in prompts:
        role = prompt["role"]
        content_text = prompt["content"]
        
        if role == "assistant":
            # Vertex AI では "assistant" ではなく "model" を使用する。
            contents.append(Content(role="model", parts=[Part.from_text(content_text)]))
        elif role == "user":
            contents.append(Content(role="user", parts=[Part.from_text(content_text)]))
    
    return contents

def generate_responses(project_id: str, location: str, model_name: str = "gemini-2.5-flash") -> None:
    """
    プロンプトデータセットを読み込み、LLM で応答を生成して、
    評価用データセットとして保存する。
    
    会話履歴全体とシステムプロンプトに対応している。
    """
    # Vertex AI 初期化する。
    init_vertexai(project_id, location)

    # 入力データを読み込む。
    dataset = get_prompt_dataset()
    results: List[EvaluationDatasetItem] = []

    print(f"Generating responses for {len(dataset)} prompts...")

    for item in dataset:
        print(f"Processing ID: {item.get('prompt_id', 'unknown')}")
        
        prompts = item.get("prompts", [])
        if not prompts:
            print("  Skipping: No prompts found.")
            continue

        # 会話履歴を構築する。
        contents = _build_conversation_contents(prompts)
        
        # モデルインスタンスを取得する（システムプロンプトを設定する）。
        model = GenerativeModel(model_name, system_instruction=SYSTEM_INSTRUCTION)
        
        # 応答を生成する。
        # 会話履歴全体を使用して応答を生成する。
        # 最後のメッセージが user であることを想定する。
        if not contents:
            print(f"  Warning: No valid conversation contents found. Skipping.")
            continue
        
        response = model.generate_content(contents)
        
        # 結果を格納する。（既存のデータ構造を維持しつつ llm_response_text を追加）
        # PromptDatasetItem を EvaluationDatasetItem に変換する。
        new_item = cast(EvaluationDatasetItem, item.copy())
        new_item["llm_response_text"] = response.text
        results.append(new_item)

    # 結果を保存する。
    save_evaluation_dataset(results)
    print(f"Saved evaluation dataset with {len(results)} items.")

