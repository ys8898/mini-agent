"""教学 Agent 的命令行入口。"""

from agent import run_agent
from request_model import request_model


def main() -> None:
    print("Tool Calling Agent 已启动。输入 /clear 清空会话，exit 或 quit 退出。")
    messages = []  # CLI 进程内的会话历史；后续 Web 接口也可按会话分别持有列表。
    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() in ("exit", "quit"):
            print("再见！")
            break
        if question.lower() == "/clear":
            messages.clear()
            print("当前会话已清空。")
            continue
        if not question:
            continue

        try:
            # CLI 只传入问题；模型请求与工具循环仍由原有模块负责。
            answer = run_agent(question, request_model, messages=messages)
        except (ValueError, RuntimeError) as exc:
            print(f"本次请求失败：{exc}")
        except Exception:
            # 意外异常不暴露内部数据，下一次用户输入仍可继续处理。
            print("本次请求发生意外错误，请重试。")
        else:
            print(f"Agent> {answer}")


if __name__ == "__main__":
    main()
