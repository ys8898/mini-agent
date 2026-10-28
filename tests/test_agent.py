"""用预设模型响应验证 Agent，不访问网络。"""

import copy
import json
import unittest
from unittest.mock import patch

from agent import run_agent


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, messages, tools):
        # 快照保留每次请求时的历史，避免后续 append 改变早先的记录。
        self.requests.append(copy.deepcopy(messages))
        self.tools = tools
        return self.replies.pop(0)


class AgentTests(unittest.TestCase):
    def test_direct_answer(self):
        model = FakeModel([{"role": "assistant", "content": "你好！"}])
        self.assertEqual(run_agent("你好", model), "你好！")
        self.assertEqual([item["role"] for item in model.requests[0]], ["system", "user"])
        self.assertEqual(len(model.tools), 2)

    def test_one_tool_then_answer(self):
        call = tool_call("call_1", "calculator", '{"expression":"2+3*4"}')
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": "结果是 14。"},
        ])
        self.assertEqual(run_agent("计算 2+3*4", model), "结果是 14。")
        history = model.requests[1]
        self.assertEqual([item["role"] for item in history], ["system", "user", "assistant", "tool"])
        self.assertEqual(history[-1]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(history[-1]["content"]), {"ok": True, "result": 14})

    def test_multiple_tools_in_one_response(self):
        calls = [
            tool_call("order_call", "get_order_status", '{"order_id":"ORD-1001"}'),
            tool_call("math_call", "calculator", '{"expression":"8/2"}'),
        ]
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "assistant", "content": "订单已发货，计算结果是 4。"},
        ])
        run_agent("查订单并计算", model, max_tool_calls=2)
        results = model.requests[1][-2:]
        assistant_calls = model.requests[1][-3]["tool_calls"]
        self.assertEqual([call["id"] for call in assistant_calls], ["order_call", "math_call"])
        self.assertEqual([item["tool_call_id"] for item in results], ["order_call", "math_call"])
        self.assertEqual(json.loads(results[0]["content"])["result"]["status"], "已发货")
        self.assertEqual(json.loads(results[1]["content"])["result"], 4)

    def test_order_status_through_agent(self):
        call = tool_call("order_1", "get_order_status", '{"order_id":"ORD-1002"}')
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": "订单正在处理中。"},
        ])
        self.assertEqual(run_agent("查询 ORD-1002", model), "订单正在处理中。")
        history = model.requests[1]
        self.assertEqual(history[-2]["tool_calls"][0]["id"], history[-1]["tool_call_id"])
        self.assertEqual(json.loads(history[-1]["content"]), {
            "ok": True,
            "result": {"order_id": "ORD-1002", "found": True, "status": "处理中"},
        })

    def test_missing_extra_and_malformed_arguments(self):
        calls = [
            tool_call("missing", "calculator", "{}"),
            tool_call("extra", "calculator", '{"expression":"1+1","unused":1}'),
            tool_call("malformed", "get_order_status", "not-json"),
            tool_call("unknown", "other_tool", "{}"),
        ]
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "assistant", "content": "参数或工具无效。"},
        ])
        self.assertEqual(run_agent("测试参数", model), "参数或工具无效。")
        results = model.requests[1][-4:]
        self.assertEqual([item["tool_call_id"] for item in results], [call["id"] for call in calls])
        for item in results:
            with self.subTest(call_id=item["tool_call_id"]):
                output = json.loads(item["content"])
                self.assertFalse(output["ok"])
                self.assertTrue(output["error"])

    def test_tool_execution_failure_is_returned_to_model(self):
        call = tool_call("failed", "calculator", '{"expression":"1+1"}')
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": "计算暂时失败。"},
        ])
        with patch("agent.calculator", side_effect=ValueError("模拟工具故障")) as calculator_mock:
            self.assertEqual(run_agent("计算 1+1", model), "计算暂时失败。")
        calculator_mock.assert_called_once_with("1+1")
        result = json.loads(model.requests[1][-1]["content"])
        self.assertEqual(result, {"ok": False, "error": "模拟工具故障"})

    def test_invalid_tool_arguments_are_returned_to_model(self):
        calls = [
            tool_call("bad_json", "calculator", "not json"),
            tool_call("zero", "calculator", '{"expression":"1/0"}'),
            tool_call("unknown", "not_allowed", "{}"),
        ]
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "assistant", "content": "工具调用失败。"},
        ])
        self.assertEqual(run_agent("测试错误", model), "工具调用失败。")
        for item in model.requests[1][-3:]:
            self.assertFalse(json.loads(item["content"])["ok"])

    def test_tool_call_limit(self):
        calls = [
            tool_call("first", "calculator", '{"expression":"1+1"}'),
            tool_call("second", "calculator", '{"expression":"2+2"}'),
        ]
        model = FakeModel([{"role": "assistant", "content": None, "tool_calls": calls}])
        with patch("agent.calculator") as calculator_mock:
            with self.assertRaisesRegex(RuntimeError, "最大工具调用次数"):
                run_agent("反复计算", model, max_tool_calls=1)
        calculator_mock.assert_not_called()
        self.assertEqual(len(model.requests), 1)

    def test_tool_call_limit_counts_across_model_requests(self):
        first = tool_call("first", "calculator", '{"expression":"1+1"}')
        second = tool_call("second", "calculator", '{"expression":"2+2"}')
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": [first]},
            {"role": "assistant", "content": None, "tool_calls": [second]},
        ])
        with self.assertRaisesRegex(RuntimeError, "最大工具调用次数"):
            run_agent("反复计算", model, max_tool_calls=1)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(len(model.requests[1]), 4)  # 第二次请求已包含第一次数学工具的结果。

    def test_bad_model_response_and_request_error(self):
        with self.assertRaisesRegex(ValueError, "最终回答"):
            run_agent("你好", FakeModel([{"role": "assistant", "content": None}]))

        def broken_model(messages, tools):
            raise OSError("网络故障")

        with self.assertRaisesRegex(RuntimeError, "模型请求失败"):
            run_agent("你好", broken_model)

    def test_follow_up_keeps_user_and_assistant_messages(self):
        messages = []
        model = FakeModel([
            {"role": "assistant", "content": "你好！"},
            {"role": "assistant", "content": "刚才我向你问好。"},
        ])
        self.assertEqual(run_agent("你好", model, messages=messages), "你好！")
        self.assertEqual(run_agent("刚才说了什么？", model, messages=messages), "刚才我向你问好。")
        self.assertEqual([item["role"] for item in model.requests[1]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(model.requests[1][2]["content"], "你好！")
        self.assertEqual([item["role"] for item in messages],
                         ["system", "user", "assistant", "user", "assistant"])

    def test_follow_up_keeps_tool_call_and_result(self):
        messages = []
        call = tool_call("order_1", "get_order_status", '{"order_id":"ORD-1001"}')
        model = FakeModel([
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": "订单已发货。"},
            {"role": "assistant", "content": "刚才查的是 ORD-1001。"},
        ])
        run_agent("查 ORD-1001", model, messages=messages)
        self.assertEqual(run_agent("刚才查的订单号？", model, messages=messages),
                         "刚才查的是 ORD-1001。")
        follow_up = model.requests[2]
        self.assertEqual([item["role"] for item in follow_up],
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(follow_up[2]["tool_calls"][0]["id"], follow_up[3]["tool_call_id"])
        self.assertEqual(json.loads(follow_up[3]["content"])["result"]["status"], "已发货")

    def test_failed_turn_does_not_change_saved_history(self):
        messages = []
        run_agent("你好", FakeModel([{"role": "assistant", "content": "你好！"}]), messages=messages)
        before = copy.deepcopy(messages)
        with self.assertRaisesRegex(RuntimeError, "模型请求失败"):
            run_agent("失败的一轮", FakeModel([]), messages=messages)
        self.assertEqual(messages, before)


if __name__ == "__main__":
    unittest.main()
