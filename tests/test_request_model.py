"""模型请求模块的离线测试：所有 HTTP 请求都由 mock 接管。"""

import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import request_model as api
from agent import run_agent


class ModelRequestTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.object(api, "load_config", return_value=(
            "test-key", "https://api.siliconflow.cn/v1", "test-model"
        ))
        self.config.start()
        self.addCleanup(self.config.stop)

    def test_request_and_response(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]}
        response = io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())
        with patch.object(api, "urlopen", return_value=response) as open_mock:
            result = api.request_model([{"role": "user", "content": "你好"}], [{"type": "function"}])
        request = open_mock.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.siliconflow.cn/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(open_mock.call_args.kwargs["timeout"], 90)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"][0]["content"], "你好")
        self.assertEqual(body["tools"], [{"type": "function"}])
        self.assertIs(body["stream"], False)
        self.assertEqual(result, message)

    def test_http_network_and_bad_json(self):
        failures = [
            (HTTPError("url", 401, "Unauthorized", {}, None), "HTTP 401"),
            (URLError("offline"), "网络连接失败"),
        ]
        for error, expected in failures:
            with self.subTest(error=expected), patch.object(api, "urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, expected):
                    api.request_model([], [])
        with patch.object(api, "urlopen", return_value=io.BytesIO(b"not-json")):
            with self.assertRaisesRegex(RuntimeError, "JSON 无法解析"):
                api.request_model([], [])

    def test_api_error_and_bad_structure(self):
        for data, expected in [
            ({"error": {"message": "private server detail"}}, "API 返回错误"),
            ({"choices": []}, "缺少 choices"),
            ({"choices": [{"message": {"role": "user"}}]}, "缺少 assistant"),
        ]:
            with self.subTest(expected=expected):
                response = io.BytesIO(json.dumps(data).encode())
                with patch.object(api, "urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, expected):
                        api.request_model([], [])

    def test_missing_config_without_reading_project_env(self):
        self.config.stop()  # 本用例需要运行真正的 load_config，而非 setUp 中的替身。
        with patch.object(api, "load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "SILICONFLOW_API_KEY"):
                api.load_config()

    def test_agent_round_trip_with_mock_http(self):
        replies = [
            {"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "calc_1", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"expression":"2+3*4"}'
                }}],
            }}]},
            {"choices": [{"message": {"role": "assistant", "content": "结果是 14。"}}]},
        ]
        sent = []

        def fake_http(request, timeout):
            sent.append(json.loads(request.data))
            return io.BytesIO(json.dumps(replies.pop(0)).encode())

        with patch.object(api, "urlopen", side_effect=fake_http):
            answer = run_agent("请计算 2+3*4", api.request_model)
        self.assertEqual(answer, "结果是 14。")
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["tools"][0]["function"]["name"], "calculator")
        tool_message = sent[1]["messages"][-1]
        self.assertEqual(tool_message["tool_call_id"], "calc_1")
        self.assertEqual(json.loads(tool_message["content"])["result"], 14)


if __name__ == "__main__":
    unittest.main()
