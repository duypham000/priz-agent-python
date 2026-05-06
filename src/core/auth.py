from dataclasses import dataclass


@dataclass
class TokenUser:
    id: str
    email: str
    username: str
    role: str
