"""
Unit tests for Safe Calculator Tool.
"""

import pytest
from app.agents.tools.calculator import evaluate_arithmetic


def test_calculator_basic_arithmetic():
    res = evaluate_arithmetic("125 * 18")
    assert res["status"] == "success"
    assert res["result"] == 2250
    assert res["formatted_result"] == "2250"


def test_calculator_percentage_calculation():
    res = evaluate_arithmetic("What is 15% of 240?")
    assert res["status"] == "success"
    assert res["result"] == 36
    assert res["formatted_result"] == "36"


def test_calculator_nested_expression():
    res = evaluate_arithmetic("(50 + 20) / 7")
    assert res["status"] == "success"
    assert res["result"] == 10
    assert res["formatted_result"] == "10"


def test_calculator_zero_division():
    res = evaluate_arithmetic("100 / 0")
    assert res["status"] == "error"
    assert "Division by zero" in res["error"]


def test_calculator_malicious_code_injection_prevention():
    # Attempt arbitrary code execution
    res1 = evaluate_arithmetic("__import__('os').system('echo pwned')")
    assert res1["status"] == "error"

    res2 = evaluate_arithmetic("open('/etc/passwd').read()")
    assert res2["status"] == "error"

    res3 = evaluate_arithmetic("exec('import sys')")
    assert res3["status"] == "error"


def test_calculator_empty_or_text_query():
    res = evaluate_arithmetic("Hello there, how are you?")
    assert res["status"] == "error"
    assert "No numbers" in res["error"]
