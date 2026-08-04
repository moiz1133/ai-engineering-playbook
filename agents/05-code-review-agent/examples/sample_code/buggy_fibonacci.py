def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)  # no memoization, O(2^n)

def get_nth_fibonacci(n):
    # Bug 1: no input validation (negative numbers cause infinite recursion)
    # Bug 2: no type checking (float input causes recursion error)
    # Bug 3: exponential time complexity for large n
    return fibonacci(n)
