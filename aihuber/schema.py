class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class Response:
    def __init__(self, content: str):
        self.content = content
