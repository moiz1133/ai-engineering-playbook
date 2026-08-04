import hashlib

def verify_password(stored_password, provided_password):
    # Bug 1: MD5 is cryptographically broken
    # Bug 2: no salt (vulnerable to rainbow table attacks)
    # Bug 3: timing attack vulnerability (string comparison)
    hashed = hashlib.md5(provided_password.encode()).hexdigest()
    return hashed == stored_password

def create_session_token(user_id):
    # Bug 1: predictable token (not cryptographically random)
    # Bug 2: no expiry embedded in token
    import time
    return f"{user_id}_{int(time.time())}"
