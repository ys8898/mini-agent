"""Agent 主循环；模型请求由调用方传入，便于离线测试。"""

import json

from tools import calculator, get_order_status


# 这是发给模型的工具说明；模型看到说明后只能提出调用请求。
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算只包含数字、括号和加减乘除的表达式。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "例如 (8-2)/3"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "查询预设的模拟订单状态。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "例如 ORD-1001"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _execute_tool(call: dict) -> str:
    """校验一次模型请求，实际执行 Python 工具，并返回 JSON 字符串。"""
    try:
        function = call["function"]
        name = function["name"]
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")

        # 名称白名单与参数检查都在 Python 侧，不能只相信模型给出的工具描述。
        if name == "calculator" and set(arguments) == {"expression"}:
            result = calculator(arguments["expression"])
        elif name == "get_order_status" and set(arguments) == {"order_id"}:
            result = get_order_status(arguments["order_id"])
        else:
            raise ValueError("未知工具或参数不匹配")
        output = {"ok": True, "result": result}
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        # 工具错误也要回传给模型，让它有机会解释或修正请求。
        output = {"ok": False, "error": str(exc)}
    return json.dumps(output, ensure_ascii=False)


def run_agent(
    question: str, request_model, max_tool_calls: int = 5, messages: list[dict] | None = None
) -> str:
    """完成一次用户提问；可选 messages 由调用方持有，以便延续会话。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题必须是非空字符串")
    if type(max_tool_calls) is not int or max_tool_calls < 1:
        raise ValueError("最大工具调用次数必须是正整数")

    # 先在副本上工作；若本轮失败，调用方保存的历史不会留下半截消息。
    history = list(messages) if messages is not None else []
    if not history:
        history.append({"role": "system", "content": "需要计算或查询订单时请使用工具，不要编造工具结果。"})
    history.append({"role": "user", "content": question})
    tool_count = 0

    while True:
        try:
            reply = request_model(history, TOOL_DEFINITIONS)
        except Exception as exc:
            raise RuntimeError("模型请求失败") from exc

        if not isinstance(reply, dict) or reply.get("role") != "assistant":
            raise ValueError("模型响应缺少 assistant 消息")
        calls = reply.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError("模型返回的 tool_calls 格式错误")
        if not calls:
            answer = reply.get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("模型没有返回最终回答")
            history.append(reply)
            if messages is not None:
                messages[:] = history
            return answer

        # 一次响应可能包含多个调用；整批检查可避免只执行了其中一部分就超限。
        if tool_count + len(calls) > max_tool_calls:
            raise RuntimeError("达到最大工具调用次数")
        history.append(reply)
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                raise ValueError("工具调用缺少有效的 id")
            output = _execute_tool(call)
            history.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            tool_count += 1
