import json
import re
from typing import Any, Dict, List, Union, Optional
from google import genai
from google.genai import types
from anthropic import AnthropicVertex

PROJECT_ID = '...'
LOCATION = '...'


def _is_gemini_model(model_name: str) -> bool:
    """モデル名が Gemini かどうかを判定する。"""
    return "gemini" in model_name.lower()


def _is_claude_model(model_name: str) -> bool:
    """モデル名が Claude かどうかを判定する。"""
    return "claude" in model_name.lower()


def _get_model(model_name: str) -> Union[genai.Client, AnthropicVertex]:
    """
    モデル名に応じて適切なモデルインスタンスを取得する。
    
    Args:
        model_name: モデル名（"gemini-2.5-pro", "gemini-2.0-flash", "claude-sonnet-4-5" など）
    
    Returns:
        genai.Client または AnthropicVertex インスタンス
    
    Raises:
        ValueError: サポートされていないモデル名が指定された場合
    """
    if _is_gemini_model(model_name):
        return genai.Client(
            vertexai=True, project=PROJECT_ID, location=LOCATION
        )
    elif _is_claude_model(model_name):
        return AnthropicVertex(
            project_id=PROJECT_ID,
            region=LOCATION,
        )
    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            f"Supported models: Gemini models (gemini-2.5-pro, gemini-2.0-flash, etc.) "
            f"or Claude models (claude-sonnet-4-5, etc.)"
        )


def _generate_json_gemini(
    model_name: str,
    prompt: str,
    schema: Dict[str, Any],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Gemini モデルを呼び出して JSON を生成する。"""
    model = _get_model(model_name)
    assert isinstance(model, genai.Client)
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema
    )
    if temperature is not None:
        config.temperature = temperature
    if max_tokens is not None:
        config.max_output_tokens = max_tokens
    
    response = model.models.generate_content(
        model=model_name,
        contents=types.Content(role="user", parts=[types.Part(text=prompt)]),
        config=config
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {response.text}")
        return None


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    テキストから JSON を抽出する。
    マークダウンのコードブロック（```json ... ```）を除去する。
    """
    text = text.strip()
    # ```json と ``` を除去する。
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text


def _generate_json_claude(
    model_name: str,
    prompt: str,
    schema: Dict[str, Any],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Claude モデルを呼び出して JSON を生成する。"""
    model = _get_model(model_name)
    assert isinstance(model, AnthropicVertex)
    
    # JSON スキーマをプロンプトに追加する。
    enhanced_prompt = prompt + f"\n\nReturn valid JSON matching this schema: {json.dumps(schema, ensure_ascii=False)}"
    
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": enhanced_prompt}]
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    response = model.messages.create(**kwargs)
    
    if not response.content:
        return None
    
    content_block = response.content[0]
    if content_block.type == 'text':
        # マークダウンのコードブロックを除去して JSON を抽出する。
        json_text = _extract_json_from_text(content_block.text)
        if json_text is None:
            print(f"Failed to extract JSON from: {content_block.text}")
            return None
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_text}")
            return None
    
    return None


def _generate_text_gemini(
    model_name: str,
    contents: List[types.Content],
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Optional[str]:
    """Gemini モデルを呼び出してテキストを生成する。"""
    model = _get_model(model_name)
    assert isinstance(model, genai.Client)
    
    config = types.GenerateContentConfig()
    if system_instruction:
        config.system_instruction = system_instruction
    if temperature is not None:
        config.temperature = temperature
    if max_tokens is not None:
        config.max_output_tokens = max_tokens
    
    response = model.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )
    return response.text


def _generate_text_claude(
    model_name: str,
    contents: List[types.Content],
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Optional[str]:
    """Claude モデルを呼び出してテキストを生成する。"""
    model = _get_model(model_name)
    assert isinstance(model, AnthropicVertex)
    
    # Content リストを Anthropic の messages 形式に変換する。
    messages = []
    for content in contents:
        if content.role == "user":
            messages.append({"role": "user", "content": content.parts[0].text if content.parts else ""})
        elif content.role == "model":
            messages.append({"role": "assistant", "content": content.parts[0].text if content.parts else ""})
    
    kwargs = {
        "model": model_name,
        "messages": messages
    }
    if system_instruction:
        kwargs["system"] = system_instruction
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    response = model.messages.create(**kwargs)
    
    if not response.content:
        return None
    
    content_block = response.content[0]
    if content_block.type == 'text':
        return content_block.text
    
    return None


def generate(
    model_name: str,
    prompt: Optional[str] = None,
    contents: Optional[List[types.Content]] = None,
    schema: Optional[Dict[str, Any]] = None,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 8192
) -> Union[str, Dict[str, Any], None]:
    """
    LLM からテキストまたは JSON を生成する。
    
    Args:
        model_name: モデル名（"gemini-2.5-pro", "gemini-2.0-flash", "claude-sonnet-4-5" など）
        prompt: プロンプト文字列（JSON 生成時に使用）
        contents: 会話履歴の Content リスト（テキスト生成時に使用）
        schema: JSON スキーマ（指定された場合は JSON 生成、None の場合はテキスト生成）
        system_instruction: システムプロンプト（オプション）
        temperature: 生成のランダム性を制御する温度パラメータ（0.0-1.0、None の場合はモデルのデフォルト値）
        max_tokens: 生成する最大トークン数（デフォルト: 8192）
    
    Returns:
        生成されたテキスト、JSON オブジェクト（辞書）、または None
    """
    if schema is not None:
        # JSON 生成モード。
        if prompt is None:
            raise ValueError("prompt is required when schema is specified")
        if _is_gemini_model(model_name):
            return _generate_json_gemini(model_name, prompt, schema, temperature, max_tokens)
        elif _is_claude_model(model_name):
            return _generate_json_claude(model_name, prompt, schema, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        # テキスト生成モード。
        if contents is None:
            raise ValueError("contents is required when schema is not specified")
        if _is_gemini_model(model_name):
            return _generate_text_gemini(model_name, contents, system_instruction, temperature, max_tokens)
        elif _is_claude_model(model_name):
            return _generate_text_claude(model_name, contents, system_instruction, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown model: {model_name}")
