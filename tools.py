"""供教学 Agent 调用的两个本地工具。"""

import ast
import math
import operator


_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ORDERS = {
    "ORD-1001": "已发货",
    "ORD-1002": "处理中",
}


def calculator(expression: str) -> int | float:
    """计算只包含数字、括号、正负号和四则运算的表达式。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式必须是非空字符串")
    if len(expression) > 200:
        raise ValueError("表达式过长")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("表达式格式错误") from exc

    def calculate(node: ast.AST, depth: int = 0) -> int | float:
        if depth > 30:
            raise ValueError("表达式嵌套过深")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = calculate(node.operand, depth + 1)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATIONS:
            left = calculate(node.left, depth + 1)
            right = calculate(node.right, depth + 1)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("不能除以零")
            value = _OPERATIONS[type(node.op)](left, right)
        else:
            raise ValueError("只支持数字和加减乘除运算")

        if not math.isfinite(value):
            raise ValueError("计算结果不是有限数字")
        return value

    return calculate(tree.body)


def get_order_status(order_id: str) -> dict[str, str | bool]:
    """从固定的模拟订单中查询状态。"""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("订单号必须是非空字符串")

    order_id = order_id.strip()
    status = _ORDERS.get(order_id)
    if status is None:
        return {"order_id": order_id, "found": False, "status": "订单不存在"}
    return {"order_id": order_id, "found": True, "status": status}
