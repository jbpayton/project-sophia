# AgentManager class is a singleton class that manages the agents in the system (only keep one instance)
from AIAgent import AIAgent
from HumanAgent import HumanAgent


class AgentManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentManager, cls).__new__(cls)
            cls._instance.agent_dict = {}

        return cls._instance

    @staticmethod
    def get_instance():
        return AgentManager()

    def get_agent_if_exists(self, agent_name):
        if agent_name in self.agent_dict:
            return self.agent_dict[agent_name]
        else:
            agent = AIAgent(agent_name)
            if agent is not None:
                self.agent_dict[agent_name] = agent
                return agent

        return None

    def send(self, agent_name, text, user_name):
        agent = self.get_agent_if_exists(agent_name)
        if agent is not None:
            return agent.send(text, user_name)

        return None, None, None, None

    @staticmethod
    def send_to_agent(agent_name, text, user_name):
        return AgentManager.get_instance().send(agent_name, text, user_name)

    @staticmethod
    def get_agent_voice(agent_name):
        agent = AgentManager.get_instance().get_agent_if_exists(agent_name)
        if agent is not None:
            return agent.profile['voice']

        return None

    @staticmethod
    def does_agent_exist(agent_name):
        return AgentManager.get_instance().get_agent_if_exists(agent_name) is not None

    @staticmethod
    def add_human_agent(agent_name):
        if agent_name in AgentManager.get_instance().agent_dict:
            return
        human_agent = HumanAgent(agent_name)
        # add to agent_dict
        AgentManager.get_instance().agent_dict[agent_name] = human_agent

    @staticmethod
    def get_queued_message(agent_name):
        agent = AgentManager.get_instance().get_agent_if_exists(agent_name)
        if agent is not None:
            # if thaa agent is a human agent, return the message
            if agent.agent_type == "Human":
                return agent.receive()

        return None

    @staticmethod
    def messages_in_queue(agent_name):
        agent = AgentManager.get_instance().get_agent_if_exists(agent_name)
        if agent is not None:
            # if this is a human agent, return the message
            if agent.agent_type == "Human":
                return agent.messages_in_queue()

        return False
