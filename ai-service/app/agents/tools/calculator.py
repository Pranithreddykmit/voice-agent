"""
Safe Calculator Tool using Abstract Syntax Tree (AST) Parsing.
Evaluates mathematical expressions without using unrestricted Python eval().
Supports standard arithmetic, percentages, and exponentiation with strict safety controls.
"""

import ast
import operator
import re
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# Allowed AST Operators mapping
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _preprocess_math_query(text: str) -> str:
    """
    Extracts and normalizes arithmetic expressions from natural language text.
    Handles 'X% of Y', 'calculate', etc.
    """
    # Replace 'X% of Y' with '(X / 100) * Y'
    pattern_pct_of = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:of|లో)?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
    text = pattern_pct_of.sub(r"(\1 / 100) * \2", text)

    # Replace standalone 'X%' with '(X / 100)'
    pattern_pct = re.compile(r"(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
    text = pattern_pct.sub(r"(\1 / 100)", text)

    # Clean leading words like 'what is', 'calculate', 'eval', etc.
    clean_text = re.sub(r"(?i)^(?:what is|calculate|evaluate|compute|find|solve|లెక్కింపు|విలువ)\s+", "", text.strip())
    # Remove trailing question marks or periods
    clean_text = clean_text.rstrip("?।.").strip()

    return clean_text


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates safe AST arithmetic nodes."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)

        # Prevent runaway power operations
        if op_type == ast.Pow:
            if right_val > 100 or left_val > 10000:
                raise ValueError("Exponent too large (safety limit: max exponent 100).")

        op_func = ALLOWED_OPERATORS[op_type]
        return op_func(left_val, right_val)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return ALLOWED_OPERATORS[op_type](operand)

    else:
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def evaluate_arithmetic(query: str) -> Dict[str, Any]:
    """
    Safely parses and evaluates an arithmetic expression from a query string.

    Returns:
        Dict with keys: expression, result, formatted_result, status, error.
    """
    expr_str = _preprocess_math_query(query)
    
    # Filter valid characters for math expression (digits, operators, parens, spaces, decimal)
    # Check if expr_str has any math characters
    if not re.search(r"\d", expr_str):
        return {
            "expression": expr_str,
            "result": None,
            "formatted_result": None,
            "status": "error",
            "error": "No numbers or arithmetic expressions found in query."
        }

    try:
        parsed = ast.parse(expr_str, mode="eval")
        result = _eval_node(parsed.body)
        
        # Format result cleanly
        if isinstance(result, float) and result.is_integer():
            result = int(result)
            formatted = str(result)
        elif isinstance(result, float):
            result = round(result, 6)
            formatted = f"{result:.4f}".rstrip("0").rstrip(".")
        else:
            formatted = str(result)

        return {
            "expression": expr_str,
            "result": result,
            "formatted_result": formatted,
            "status": "success",
            "error": None
        }

    except ZeroDivisionError:
        return {
            "expression": expr_str,
            "result": None,
            "formatted_result": None,
            "status": "error",
            "error": "Division by zero is mathematically undefined."
        }
    except Exception as e:
        logger.warning(f"Calculator evaluation error on '{expr_str}': {str(e)}")
        return {
            "expression": expr_str,
            "result": None,
            "formatted_result": None,
            "status": "error",
            "error": f"Invalid or unsafe arithmetic expression: {str(e)}"
        }
