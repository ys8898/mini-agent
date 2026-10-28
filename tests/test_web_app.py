"""HTTP 接口离线测试，模型请求全部使用假响应。"""

import copy
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        with web_app._registry_lock:
            web_app._sessions.clear()
        self.client = TestClient(web_app.app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_and_follow_up(self):
        requests = []
        replies = [
            {"role": "assistant", "content": "你好！"},
            {"role": "assistant", "content": "刚才我说了你好。"},
        ]

        def fake_model(messages, tools):
            requests.append(copy.deepcopy(messages))
            return replies.pop(0)

        with patch("web_app.request_model", side_effect=fake_model):
            first = self.client.post("/chat", json={"message": "你好"})
            second = self.client.post("/chat", json={
                "message": "刚才你说了什么？", "session_id": first.json()["session_id"]
            })
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["session_id"], second.json()["session_id"])
        self.assertEqual(second.json()["answer"], "刚才我说了你好。")
        self.assertEqual([item["role"] for item in requests[1]],
                         ["system", "user", "assistant", "user"])

    def test_different_sessions_are_isolated(self):
        requests = []

        def fake_model(messages, tools):
            requests.append(copy.deepcopy(messages))
            return {"role": "assistant", "content": "收到"}

        with patch("web_app.request_model", side_effect=fake_model):
            a = self.client.post("/chat", json={"message": "甲", "session_id": "a"})
            b = self.client.post("/chat", json={"message": "乙", "session_id": "b"})
        self.assertEqual((a.status_code, b.status_code), (200, 200))
        self.assertEqual([item["role"] for item in requests[1]], ["system", "user"])
        self.assertEqual(requests[1][1]["content"], "乙")

    def test_invalid_request_and_model_error(self):
        for body in ({}, {"message": ""}, {"message": "  "},
                     {"message": "你好", "session_id": "  "},
                     {"message": "你好", "extra": 1}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/chat", json=body).status_code, 422)

        with patch("web_app.request_model", side_effect=OSError("模拟网络故障")):
            response = self.client.post("/chat", json={"message": "你好", "session_id": "failed"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Agent 或模型请求失败")
        self.assertEqual(web_app._sessions["failed"].messages, [])

    def test_same_session_requests_are_serialized(self):
        active = 0
        peak = 0
        guard = threading.Lock()

        def fake_model(messages, tools):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return {"role": "assistant", "content": "收到"}

        with patch("web_app.request_model", side_effect=fake_model):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(
                    lambda text: self.client.post("/chat", json={"message": text, "session_id": "shared"}),
                    ("第一问", "第二问"),
                ))
        self.assertEqual([response.status_code for response in results], [200, 200])
        self.assertEqual(peak, 1)
        self.assertEqual(len(web_app._sessions["shared"].messages), 5)


if __name__ == "__main__":
    unittest.main()
