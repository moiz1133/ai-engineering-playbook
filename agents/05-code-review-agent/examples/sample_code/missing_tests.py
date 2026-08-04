def divide(a, b):
    return a / b  # Bug: ZeroDivisionError unhandled

def parse_config(config_str: str) -> dict:
    # Bug: no validation, no error handling
    import json
    return json.loads(config_str)

def calculate_discount(price: float, discount_pct: float) -> float:
    # Bug: no validation (negative price, discount > 100% allowed)
    return price * (1 - discount_pct / 100)
