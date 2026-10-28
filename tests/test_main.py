"""CLI 离线测试：替换输入和 Agent，避免真实 API 调用。"""

import io
import copy
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main


class MainTests(unittest.TestCase):
    def test_multiple_questions_and_exit(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=[" 你好 ", "  ", "查订单", "EXIT"]), \
                patch("main.run_agent", side_effect=["你好！", "订单已发货。"]) as agent_mock, \
                redirect_stdout(output):
            main.main()
        self.assertEqual(agent_mock.call_count, 2)
        self.assertEqual(agent_mock.call_args_list[0].args, ("你好", main.request_model))
        self.assertEqual(agent_mock.call_args_list[1].args, ("查订单", main.request_model))
        self.assertIs(agent_mock.call_args_list[0].kwargs["messages"],
                      agent_mock.call_args_list[1].kwargs["messages"])
        self.assertIn("Agent> 你好！", output.getvalue())
        self.assertIn("Agent> 订单已发货。", output.getvalue())

    def test_error_does_not_stop_next_question(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["第一问", "第二问", "quit"]), \
                patch("main.run_agent", side_effect=[RuntimeError("模型请求失败"), "第二问的回答"]) as agent_mock, \
                redirect_stdout(output):
            main.main()
        self.assertEqual(agent_mock.call_count, 2)
        self.assertIn("本次请求失败：模型请求失败", output.getvalue())
        self.assertIn("Agent> 第二问的回答", output.getvalue())

    def test_eof_and_keyboard_interrupt_exit_cleanly(self):
        for signal in (EOFError(), KeyboardInterrupt()):
            with self.subTest(signal=type(signal).__name__):
                output = io.StringIO()
                with patch("builtins.input", side_effect=signal), \
                        patch("main.run_agent") as agent_mock, redirect_stdout(output):
                    main.main()
                agent_mock.assert_not_called()
                self.assertIn("再见！", output.getvalue())

    def test_clear_starts_new_history(self):
        replies = [
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "assistant", "content": "清空后的回答"},
        ]
        requests = []

        def fake_model(messages, tools):
            requests.append(copy.deepcopy(messages))
            return replies.pop(0)

        output = io.StringIO()
        with patch("builtins.input", side_effect=["第一问", "/clear", "第二问", "quit"]), \
                patch("main.request_model", side_effect=fake_model), redirect_stdout(output):
            main.main()
        self.assertEqual([item["role"] for item in requests[1]], ["system", "user"])
        self.assertEqual(requests[1][1]["content"], "第二问")
        self.assertIn("当前会话已清空。", output.getvalue())


if __name__ == "__main__":
    unittest.main()
