import queue


class HumanAgent:
    def __init__(self, name):
        self.name = name
        self.message_queue = queue.Queue()
        self.agent_type = "Human"

    def send(self, message, sender_name="Agent"):
        self.message_queue.put({"sender": sender_name, "content": message})

    def receive(self):
        if not self.message_queue.empty():
            return self.message_queue.get()
        else:
            return None

    def messages_in_queue(self):
        return not self.message_queue.empty()
