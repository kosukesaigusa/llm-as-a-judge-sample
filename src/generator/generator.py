from typing import List, cast, Literal
from google.genai import types
from src.data import get_prompt_dataset, save_evaluation_dataset
from src.models import generate
from src.types import EvaluationDatasetItem, PromptItem
from .prompt import SYSTEM_INSTRUCTION_GOOD, SYSTEM_INSTRUCTION_POOR

def _build_conversation_contents(prompts: List[PromptItem]) -> List[types.Content]:
    """
    プロンプトリストから会話履歴の Content リストを構築する。
    
    Returns:
        contents: Content リスト
    """
    contents: List[types.Content] = []
    
    for prompt in prompts:
        role = prompt["role"]
        content_text = prompt["content"]
        
        if role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content_text)]))
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=content_text)]))
    
    return contents

def generate_responses(model_name: str,quality: Literal["good", "poor"]) -> None:
    """
    プロンプトデータセットを読み込み、LLM で応答を生成して、
    評価用データセットとして保存する。
    
    会話履歴全体とシステムプロンプトに対応している。
    
    Args:
        model_name: 使用するモデル名
        quality: 応答の品質（"good" または "poor"）
    """
    # システムプロンプトを選択する。
    if quality == "good":
        system_instruction = SYSTEM_INSTRUCTION_GOOD
    else:
        system_instruction = SYSTEM_INSTRUCTION_POOR

    # 入力データを読み込む。
    dataset = get_prompt_dataset()
    results: List[EvaluationDatasetItem] = []

    print(f"Generating responses for {len(dataset)} prompts (quality: {quality})...")

    for i, item in enumerate(dataset):
        print(f"Processing item {i+1}/{len(dataset)}")
        
        prompts = item.get("prompts", [])
        if not prompts:
            print("  Skipping: No prompts found.")
            continue

        # 会話履歴を構築する。
        contents = _build_conversation_contents(prompts)
        
        # 応答を生成する。
        if not contents:
            print(f"  Warning: No valid conversation contents found. Skipping.")
            continue
        
        result = generate(
            model_name=model_name,
            contents=contents,
            system_instruction=system_instruction
        )
        
        if not result or not isinstance(result, str):
            print(f"  Warning: Failed to generate response. Skipping.")
            continue
        
        # 結果を格納する（既存のデータ構造を維持しつつ llm_response_text を追加する）。
        new_item = cast(EvaluationDatasetItem, item.copy())
        new_item["llm_response_text"] = result
        results.append(new_item)

    # 結果を保存する。
    saved_path = save_evaluation_dataset(results)
    print(f"Saved evaluation dataset with {len(results)} items to: {saved_path}")
