import json

from GraphStore import GraphStore
from langchain.schema import SystemMessage, HumanMessage


class GraphStoreMemory():
    def __init__(self, model):
        self.current_topic = "<nothing yet>"
        self.topics = []
        self.relevant_entities = ["<not yet set>"]
        self.entities_db = GraphStore("entities")
        self.model = model
        self.name = "CogDBGraphStore"

    def get_graph_from_conversation(self, message_history):
        build_topic_graphs_prompt = "You are an AI who reads conversations and provides lists of topics and triples " \
                                    "to make up knowledge graphs based " \
                                    "on entities discussed in the conversation and the relationships between them. " \
                                    "The graph should be built from RDF triples of the form (subject, predicate, " \
                                    "object). For example,  (I, like, apples) or (Bob, is a father to, Ann). "

        request_prompt = "\nPlease get a list of topics (high level conversation topics), the current topic, and the " \
                         "RDF triples (subject, predicate, object) from this conversation. " \
                         "Also, take care to not add duplicate entries. " \
                         "Please avoid capturing actions and stick more to describing the entities via " \
                         "relationships. Also get the current topic and 5 most relevant entities " \
                         "to the conversation. The previously determined current topic was " + self.current_topic + \
                         " and the relevant entities were " + ",".join(self.relevant_entities) + "\n"

        message = self.model(
            [
                SystemMessage(role=self.name, content=build_topic_graphs_prompt),
                HumanMessage(content=request_prompt + ":\n".join(message_history)),
            ]
        )
        print("Got message: " + message.content)
        message = self.model(
            [
                SystemMessage(role=self.name, content=build_topic_graphs_prompt),
                HumanMessage(content="\nPlease format this in in JSON, in this format:"
                                     "{"
                                     " \"topics\": [\r\n"
                                     "    \"topic 1\",\r\n"
                                     "    \"topic 2\",\r\n"
                                     " ],\r\n"
                                     " \"current_topic\": \"topic\",\r\n"
                                     " \"entities\": [\r\n"
                                     "    \"entity 1\",\r\n"
                                     "    \"entity 2\",\r\n"
                                     " ],\r\n"
                                     "  \"triples\": [\r\n"
                                     "    {\r\n"
                                     "      \"subject\": \"Sophia\",\r\n"
                                     "      \"predicate\": \"cares about\",\r\n"
                                     "      \"object\": \"Joey\"\r\n"
                                     "    },..."
                                     " ],\r\n"
                                     " \nTriples should link attributes and relationships to the subject, "
                                     "rather than simply stating what is said:\n" + message.content),
            ]
        )

        print("Got message: " + message.content)

        # Parse the message.content and add to the graph
        data = json.loads(message.content)
        for triple in data['triples']:
            subject = triple['subject']
            predicate = triple['predicate']
            obj = triple['object']
            self.entities_db.add_edge(subject, predicate, obj)

        self.topics = data['topics']
        self.current_topic = data['current_topic']
        self.relevant_entities = data['entities']

        print("Into network")

    def get_current_topic(self):
        return self.current_topic

    def get_topics(self):
        return self.topics

    def get_relevant_entities(self):
        return self.relevant_entities

    def simplify_graph(self):

        network_string = self.get_network_for_relevant_entities()
        build_topic_graphs_prompt = "You are a tool to prune and simplify knowledge graphs. " \
                                    "You are careful not to destroy data, while removing unnecessary " \
                                    "entities and predicates, as well as merging what needs to be merged. Also, " \
                                    "remove inconsistent or redundant links" \

        message = self.model(
            [
                SystemMessage(role=self.name, content=build_topic_graphs_prompt),
                HumanMessage(content="\nPlease format this in in JSON, in this format (for example):"
                                     "{"
                                     " \"entities_to_remove\": [\r\n"
                                     "    \"entity 1\",\r\n"
                                     "    \"entity 2\",\r\n"
                                     " ],\r\n"
                                     " \"predicates_to_remove\": [\r\n"
                                     "    \"entity 1\",\r\n"
                                     "    \"entity 2\",\r\n"
                                     " ],\r\n"
                                     "  \"entities_to_merge\": [\r\n"
                                     "    {\r\n"
                                     "      \"id_to_keep\": \"Joey\",\r\n"
                                     "      \"id_to_merge\": \"Joseph\",\r\n"
                                     "    },..."
                                     " ],\r\n"
                                     "  \"predicates_to_merge\": [\r\n"
                                     "    {\r\n"
                                     "      \"id_to_keep\": \"Joey\",\r\n"
                                     "      \"id_to_merge\": \"Joseph\",\r\n"
                                     "    },..."
                                     " ],\r\n"
                                     "  \"links_to_remove\": [\r\n"
                                     "    {\r\n"
                                     "      \"id_1\": \"Joey\",\r\n"
                                     "      \"predicate\": \"is\",\r\n"
                                     "      \"id_2\": \"Joey\",\r\n"
                                     "    },..."
                                     " ],\r\n"
                                     " \nThe network is as follows:\n" + network_string),
            ]
        )
        print(message.content)
        self.entities_db.process_instructions(message.content)

    def get_network_for_relevant_entities(self, simplify=False):
        entity_network_str = ""

        for entity in self.relevant_entities:
            entity_str = self.entities_db.get_network_string(entity)
            entity_network_str += entity_str + "\n"  # Add an empty line between entities for clarity

        if simplify:
            self.simplify_graph()
            entity_network_str = self.get_network_for_relevant_entities()

        return entity_network_str

