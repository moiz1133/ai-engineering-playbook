def find_users_by_department(users: list, department: str) -> list:
    # Bug 1: O(n) linear scan on every call -- no indexing
    # Bug 2: no input validation on department string
    # Bug 3: returns mutable internal list reference
    result = []
    for user in users:
        if user['department'] == department:
            result.append(user)
    return result

def get_user_email(users: list, user_id: int) -> str:
    # Bug 1: no handling of user not found (returns None implicitly)
    # Bug 2: O(n) scan instead of dict lookup
    for user in users:
        if user['id'] == user_id:
            return user['email']
