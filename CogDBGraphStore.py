from cog.torque import Graph
from langchain.schema import SystemMessage, HumanMessage


class CogDBGraphStore():
    def __init__(self, model):
        self.topics_db = Graph("topics")
        self.entities_db = Graph("entities")
        self.model = model
        self.name = "CogDBGraphStore"

    def get_graph_from_conversation(self, message_history):
        build_topic_graphs_prompt = "You are an AI who reads conversations and builds graphs of topics and entities. " \
                                    "The graph should be built from triples of the form (subject, predicate, " \
                                    "object). For example, the sentence 'I like apples' would be represented as (I, " \
                                    "like, apples). "

        print(f"{self.name}: ")
        message = self.model(
            [
                SystemMessage(role=self.name, content=build_topic_graphs_prompt),
                HumanMessage(content="\nWould you be able to get a list of topics and entity triples from this conversation?:\n".join(message_history)),
            ]
        )
        print(message.content)

