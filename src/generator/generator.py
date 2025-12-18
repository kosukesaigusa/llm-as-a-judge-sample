from google.genai import types

from src.data import get_generation_dataset, get_rubrics, save_evaluation_dataset
from src.models import generate
from src.types import EvaluationDatasetItem, PromptItem


def _build_conversation_contents(prompts: list[PromptItem]) -> list[types.Content]:
    """
    プロンプトリストから会話履歴の Content リストを構築する。

    Returns:
        contents: Content リスト
    """
    contents: list[types.Content] = []

    for prompt in prompts:
        role = prompt["role"]
        content_text = prompt["content"]

        if role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content_text)]))
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=content_text)]))

    return contents


def generate_responses(model_name: str) -> None:
    """
    生成用データセットを読み込み、LLM で応答を生成して、
    評価用データセットとして保存する。

    会話履歴全体とシステムプロンプトに対応している。
    各アイテムの generator_system_instructions ごとに応答を生成する。

    Args:
        model_name: 使用するモデル名
    """
    # 入力データを読み込む。
    dataset = get_generation_dataset()
    rubrics = get_rubrics()
    results: list[EvaluationDatasetItem] = []

    print(f"Generating responses for {len(dataset)} items...")

    for i, item in enumerate(dataset):
        print(f"Processing item {i + 1}/{len(dataset)}")

        prompts = item.get("prompts", [])
        system_instructions = item.get("generator_system_instructions", [])

        if not prompts:
            print("  Skipping: No prompts found.")
            continue

        if not system_instructions:
            print("  Skipping: No system instructions found.")
            continue

        # 会話履歴を構築する。
        contents = _build_conversation_contents(prompts)

        # 応答を生成する。
        if not contents:
            print("  Warning: No valid conversation contents found. Skipping.")
            continue

        # 各システム指示に対して応答を生成する。
        for j, system_instruction in enumerate(system_instructions):
            print(f"  Generating response for instruction {j + 1}/{len(system_instructions)}")

            result = generate(model_name=model_name, contents=contents, system_instruction=system_instruction)

            if not result or not isinstance(result, str):
                print(f"  Warning: Failed to generate response for instruction {j + 1}. Skipping.")
                continue

            # 結果を格納する。
            new_item: EvaluationDatasetItem = {"prompts": prompts, "rubrics": rubrics, "llm_response_text": result}
            results.append(new_item)

    # 結果を保存する。
    save_evaluation_dataset(results)
    print(f"Saved evaluation dataset with {len(results)} items.")
