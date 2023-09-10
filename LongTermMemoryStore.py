import json
import time
import os
from datetime import datetime, timedelta

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from GraphStore import GraphStore
from langchain.schema import SystemMessage, HumanMessage
import threading


class ConversationFileLogger:
    def __init__(self, directory="logs"):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_log_file_path(self, date_str):
        return os.path.join(self.directory, f"{date_str}.txt")

    def log_tool_output(self, tool_name, output):
        # create a directory for the tool if it doesn't exist
        tool_directory = os.path.join(self.directory, tool_name)
        if not os.path.exists(tool_directory):
            os.makedirs(tool_directory)
            # create a timestamped file for the output (one output per file)
        timestamp = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        file_path = os.path.join(tool_directory, f"{timestamp}.txt")
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(output)
        # return the path to the file
        return file_path

    def log_message(self, message_to_log):
        # Write the message to the log file
        date_str = time.strftime("%Y-%m-%d", time.localtime())
        with open(self.get_log_file_path(date_str), 'a', encoding="utf-8") as f:
            f.write(message_to_log + '\n')

    def load_last_n_lines(self, n):
        lines_to_return = []
        current_date = datetime.now()
        while n > 0 and current_date > datetime(2000, 1, 1):  # Assuming logs won't be from before the year 2000
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = self.get_log_file_path(date_str)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) <= n:
                        lines_to_return = lines + lines_to_return
                        n -= len(lines)
                    else:
                        lines_to_return = lines[-n:] + lines_to_return
                        n = 0
            current_date -= timedelta(days=1)

        return lines_to_return


class LongTermMemoryStore:
    def __init__(self, model, agent_name=""):
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
        self.conversation_logger = ConversationFileLogger(agent_name + "_logs")

    def accept_message(self, message):
        self.message_buffer.append(message)
        self.conversation_logger.log_message(message)
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

    def update_graph_from_conversation(self, conversation_buffer):
        build_topic_graphs_prompt = f"""You are an AI who reads conversations and builds knowledge graphs based on the 
        entities discussed and the relationships between them. These knowledge graphs are constructed from RDF 
        triples of the form (subject, predicate, object), e.g., (I, like, apples) or (Bob, is a father to, Ann). 

        To ensure the consistency and reusability of the knowledge graphs, consider the following guidelines while 
        identifying the RDF triples: 

        1. **Verb Form**: Use consistent verb forms (preferably base form) for predicates, e.g., "support", "love", 
        "dislike". 2. **Specific Relationships**: Identify specific, directional verbs that clearly indicate the 
        nature of the relationship between the subject and the object, e.g., "is a parent of", "works at". 3. 
        **Controlled Vocabulary**: Stick to a predefined list of predicates to describe relationships, 
        and map similar predicates to a standard term to maintain consistency. 4. **Hierarchical Relationships**: 
        Where possible, create hierarchical relationships between predicates to group similar relationships together. 
        5. **NLP Techniques**: Utilize techniques such as lemmatization to reduce predicates to their base form, 
        and employ NLP techniques to extract entities and relationships more accurately from the conversation. 

        Your task is to identify a list of topics (high-level conversation topics), the current topic, and the RDF 
        triples from this conversation. Also, avoid adding duplicate entries and focus on describing entities through 
        relationships rather than capturing actions. 

        Additionally, identify the current topic and the 5 most relevant entities to the conversation. The previously 
        determined current topic was "{self.current_topic}" and the relevant entities were "{', '.join(self.relevant_entities)}". 

        Please proceed with generating the knowledge graph based on the conversation provided.
        """

        request_prompt = "\nPlease get a list of topics (high level conversation topics), the current topic, and the " \
                         "RDF triples (subject, predicate, object) from this conversation. " \
                         "Also, take care to not add duplicate entries. Also get the current topic and 5 most " \
                         "relevant entities to the conversation."

        message = self.model(
            [
                SystemMessage(role=self.name, content=build_topic_graphs_prompt),
                HumanMessage(content=request_prompt + ":\n".join(conversation_buffer)),
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
                                     "      \"subject\": \"entity 1\",\r\n"
                                     "      \"predicate\": \"predicate\",\r\n"
                                     "      \"object\": \"entity 2\"\r\n"
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
