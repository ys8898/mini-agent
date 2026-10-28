"""通过标准库请求硅基流动的 Chat Completions 接口。"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def load_config() -> tuple[str, str, str]:
    """加载本项目的 .env；返回密钥、接口根地址、模型名。"""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    base_url = os.getenv("SILICONFLOW_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("SILICONFLOW_MODEL", "").strip()
    missing = [name for name, value in (
        ("SILICONFLOW_API_KEY", api_key),
        ("SILICONFLOW_BASE_URL", base_url),
        ("SILICONFLOW_MODEL", model),
    ) if not value]
    if missing:
        raise ValueError("缺少配置：" + ", ".join(missing))
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path != "/v1":
        raise ValueError("SILICONFLOW_BASE_URL 应为以 /v1 结尾的 HTTPS 地址")
    return api_key, base_url, model


def request_model(messages: list[dict], tools: list[dict]) -> dict:
    """发送对话历史和工具定义，返回第一条 assistant 消息。"""
    api_key, base_url, model = load_config()
    payload = {"model": model, "messages": messages, "tools": tools, "stream": False}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        base_url + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )

    try:
        # 超时避免网络请求无限等待；不把请求头或服务端原始错误写入异常。
        with urlopen(request, timeout=90) as response:
            data = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"模型 API 返回 HTTP {exc.code}") from None
    except URLError:
        raise RuntimeError("模型 API 网络连接失败") from None
    except (UnicodeError, json.JSONDecodeError):
        raise RuntimeError("模型 API 返回的 JSON 无法解析") from None

    if not isinstance(data, dict):
        raise RuntimeError("模型 API 响应格式错误")
    if data.get("error"):
        raise RuntimeError("模型 API 返回错误，请检查模型、额度和服务状态")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("模型 API 响应缺少 choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RuntimeError("模型 API 响应缺少 assistant 消息")
    return message
