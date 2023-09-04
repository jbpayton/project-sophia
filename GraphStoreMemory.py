import json

from GraphStore import GraphStore
from langchain.schema import SystemMessage, HumanMessage
import threading


class GraphStoreMemory:
    def __init__(self, model):
        self.thread_lock = threading.Lock()
        self.message_buffer = []
        self.graph_processing_size = 10  # number of messages to process at a time
        self.graph_processing_overlap = 5  # number of messages to overlap between processing
        self.current_topic = "<nothing yet>"
        self.topics = []
        self.relevant_entities = ["<not yet set>"]
        self.entities_db = GraphStore("entities")
        self.model = model
        self.name = "GraphStoreMemory"

    def accept_message(self, message):
        self.message_buffer.append(message)
        if len(self.message_buffer) >= self.graph_processing_size:
            # create a copy of the message buffer to process
            message_buffer_copy = self.message_buffer.copy()
            self.message_buffer = self.message_buffer[-self.graph_processing_overlap:]

            # process the batch of messages in a separate thread (as a one shot daemon)
            threading.Thread(target=self.process_buffer, args=(message_buffer_copy,), daemon=True).start()

    def process_buffer(self, message_buffer):
        # lock the thread
        self.thread_lock.acquire()
        print("\nStarting to process a batch of messages")
        try:
            # get the graph from the conversation
            self.update_graph_from_conversation(message_buffer)
            network_string = self.get_network_for_relevant_entities(simplify=True)
            print("Network string: " + network_string)
        finally:
            # release the lock
            self.thread_lock.release()
            print("\nFinished processing a batch of messages")

    def update_graph_from_conversation(self, message_history):
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
            try:
                subject = triple['subject']
                predicate = triple['predicate']
                obj = triple['object']
            except KeyError:
                print("Error: Triple missing subject, predicate, or object")
                continue
            self.entities_db.add_edge(subject, predicate, obj)

        self.topics = data['topics']
        self.current_topic = data['current_topic']
        self.relevant_entities = data['entities']

        self.entities_db.save_to_file()

    def get_current_topic(self):
        return self.current_topic

    def get_topics(self):
        return self.topics

    def get_relevant_entities(self):
        return self.relevant_entities

    def simplify_graph(self):

        network_string = self.get_network_for_relevant_entities()
        build_topic_graphs_prompt = 'You are a tool designed to prune and simplify conversational knowledge graphs. ' \
                                    'Your goal is to meticulously remove duplicate entities and predicates that ' \
                                    'don\'t contribute to the understanding of a conversation, while preserving ' \
                                    'valuable data. If predicates are synonymous, consolidate them. Remove ' \
                                    'inconsistent links, but take care with redundant ones as they can serve as ' \
                                    'backups. It\'s crucial, however, to respect the individuality of distinct ' \
                                    'entities involved in the conversation, like "Sophia" and "Joey". They should ' \
                                    'remain separate to maintain the integrity of the conversation. Please ensure ' \
                                    'that you don\'t destroy too many precious memories of the agent as they are ' \
                                    'needed for optimal function. Once you have made your pruning decisions, ' \
                                    'present them for validation before final application to prevent any unintended ' \
                                    'alterations. '
        message = self.model(
            [
                SystemMessage(role=self.name, content=build_topic_graphs_prompt),
                HumanMessage(
                    content="\nPlease format this in JSON, in this format (this is just an example). You can add more "
                            "items to each list as needed: "
                            "{"
                            " \"entities_to_remove\": [\r\n"
                            "    \"entity 1\",\r\n"
                            "    \"entity 2\"\r\n"
                            " ],\r\n"
                            " \"predicates_to_remove\": [\r\n"
                            "    \"entity 1\",\r\n"
                            "    \"entity 2\"\r\n"
                            " ],\r\n"
                            "  \"entities_to_merge\": [\r\n"
                            "    {\r\n"
                            "      \"id_to_keep\": \"entity 3\",\r\n"
                            "      \"id_to_merge\": \"entity 4\"\r\n"
                            "    }\r\n"
                            " ],\r\n"
                            "  \"predicates_to_merge\": [\r\n"
                            "    {\r\n"
                            "      \"id_to_keep\": \"predicate_1\",\r\n"
                            "      \"id_to_merge\": \"predicate_2\"\r\n"
                            "    }\r\n"
                            " ],\r\n"
                            "  \"links_to_remove\": [\r\n"
                            "    {\r\n"
                            "      \"id_1\": \"entity 6\",\r\n"
                            "      \"predicate\": \"predicate 56\",\r\n"
                            "      \"id_2\": \"entity 7\"\r\n"
                            "    }\r\n"
                            " ]\r\n"
                            "}\n\nThe network is as follows:\n" + network_string),

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
