"""本地工具的离线测试。"""

import unittest

from tools import calculator, get_order_status


class CalculatorTests(unittest.TestCase):
    def test_four_operations_and_parentheses(self):
        self.assertEqual(calculator("2 + 3 * 4"), 14)
        self.assertEqual(calculator("(8 - 2) / 3"), 2)
        self.assertEqual(calculator("-5 + +2"), -3)

    def test_invalid_expression(self):
        for expression in ("", "2 +", "2 ** 3", "abs(-2)", "True + 1", "1e309"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculator(expression)

    def test_invalid_type_and_zero_division(self):
        with self.assertRaises(ValueError):
            calculator(None)
        with self.assertRaises(ZeroDivisionError):
            calculator("10 / (3 - 3)")


class OrderTests(unittest.TestCase):
    def test_existing_orders(self):
        self.assertEqual(
            get_order_status("ORD-1001"),
            {"order_id": "ORD-1001", "found": True, "status": "已发货"},
        )
        self.assertEqual(get_order_status("ORD-1002")["status"], "处理中")

    def test_missing_order(self):
        self.assertEqual(
            get_order_status("ORD-9999"),
            {"order_id": "ORD-9999", "found": False, "status": "订单不存在"},
        )

    def test_invalid_order_id(self):
        for order_id in ("", "  ", None):
            with self.subTest(order_id=order_id):
                with self.assertRaises(ValueError):
                    get_order_status(order_id)


if __name__ == "__main__":
    unittest.main()
