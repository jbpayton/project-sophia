import json
import time
import os
from datetime import datetime, timedelta

from VectorKnowledgeGraph import VectorKnowledgeGraph

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
        self.knowledge_store = VectorKnowledgeGraph(path="GraphStoreMemory")
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
            network_string = self.get_network_for_relevant_entities()
            print("Network string: " + network_string)
        finally:
            # release the lock
            self.thread_lock.release()
            print("\nFinished processing a batch of messages")

    def update_graph_from_conversation(self, conversation_buffer):
        # create a string from the conversation buffer using join
        conversation_string = "\n".join(conversation_buffer)

        # Prepare metadata for the graph
        # Get the current timestamp
        timestamp = datetime.now().isoformat()
        metadata = {'timestamp': timestamp, "reference": "conversation"}

        # update the graph from the conversation string
        self.knowledge_store.process_text(conversation_string, metadata=metadata)
        self.knowledge_store.save()

    def get_current_topic(self):
        return self.current_topic

    def get_topics(self):
        return self.topics

    def get_relevant_entities(self):
        return self.relevant_entities

    def get_network_for_relevant_entities(self, simplify=False):
        entity_network_str = ""

        for entity in self.relevant_entities:
            entity_str = self, self.knowledge_store.build_graph_from_noun(entity)
            #entity_network_str += entity_str + "\n"  # Add an empty line between entities for clarity

        return entity_network_str
