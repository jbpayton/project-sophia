import json
import re
import shutil
import time
from functools import reduce
import faiss
import os
from matplotlib import pyplot as plt

import networkx as nx
from tinydb import TinyDB, Query

import numpy as np

from util import load_secrets
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from ContextSummarizers import summarize_messages_tuples, msgs2string

# this is used in place of an alternate map function, allowing for the specification of alternative map functions
class IdentityMap:
    def __getitem__(self, key):
        return key

class VectorKnowledgeGraph:
    def __init__(self, chat_llm=None, embedding_model=None, embedding_dim=384, path="VectorKnowledgeGraphData"):
        if chat_llm is None:
            # this may have been loaded earlier / somewhere else, but we need to make sure it's loaded before we use it

            self.client = OpenAI(
                api_key="sk-111111111111111111111111111111111111111111111111",
                base_url=os.environ['LOCAL_TEXTGEN_API_BASE']
            )
        else:
            self.client = chat_llm

        if embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            self.embedding_model = embedding_model
            self.embedding_dim = embedding_dim

        self.triples_list = []
        self.id_to_triple = {}
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

        self.save_path = path
        if not self.load(path):
            os.makedirs(path, exist_ok=True)  # Create directory if not exists

        # Initialize TinyDB
        self.metadata_db = TinyDB(f'{path}/db.json')

        # create a folder to store raw messages
        self.raw_messages_folder = f'{path}/raw_messages'
        os.makedirs(self.raw_messages_folder, exist_ok=True)

    def save(self, path=""):
        try:
            if path == "":
                path = self.save_path

            os.makedirs(path, exist_ok=True)  # Create directory if not exists

            # Convert triples from lists to tuples, if necessary
            id_to_triple_tuple = {k: tuple(v) for k, v in self.id_to_triple.items()}

            # Save triples_list and id_to_triple to JSON
            with open(f'{path}/triples_list.json', 'w') as f:
                json.dump([tuple(triple) for triple in self.triples_list], f)  # Convert triples to tuples
            with open(f'{path}/id_to_triple.json', 'w') as f:
                json.dump({str(k): v for k, v in id_to_triple_tuple.items()}, f)  # Convert keys to strings

            # Save faiss_index to binary file
            faiss.write_index(self.faiss_index, f'{path}/faiss_index.bin')
        except Exception as e:
            print(f"Error saving data: {e}")

    def load(self, path="VectorKnowledgeGraphData"):
        try:
            # Reset the data
            self.triples_list = []
            self.id_to_triple = {}
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

            # Load triples_list and id_to_triple from JSON
            with open(f'{path}/triples_list.json', 'r') as f:
                self.triples_list = [tuple(triple) for triple in json.load(f)]
            with open(f'{path}/id_to_triple.json', 'r') as f:
                self.id_to_triple = {int(k): tuple(v) for k, v in json.load(f).items()}  # Convert keys back to integers

            # Load faiss_index from binary file
            self.faiss_index = faiss.read_index(f'{path}/faiss_index.bin')
            return True  # Return True if loading succeeded
        except Exception as e:
            print(f"Error loading data: {e}")
            # Reset the data if loading failed, we don't want an incoherent state
            self.triples_list = []
            self.id_to_triple = {}
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            return False  # Return False if loading failed

    def process_text(self, input_text, metadata=None, batch_size=50):
        # if the input text is a list of messages, concatenate them into a single string
        if isinstance(input_text, list):
            # check to see if input text is a list of dictionaries
            if isinstance(input_text[0], dict):
                # get each 'content' key from the list of dictionaries and maka e a list of messages
                sentences = [msg['content'] for msg in input_text]
            else:
                sentences = input_text
            input_text = msgs2string(input_text, start_index=0, end_index=-1)
        else:
            # separate the text into sentences
            sentences = self.split_sentences(input_text)

        if metadata is None:
            metadata = {}

        # create a timestamp for the metadata
        metadata['timestamp'] = time.time()

        filename = f'{self.raw_messages_folder}/{metadata["timestamp"]}.txt'

        #if there is a name collision, add a number to the filename
        file_number = 0
        while os.path.exists(filename):
            metadata['timestamp'] = time.time()
            filename = f'{self.raw_messages_folder}/{metadata["timestamp"]}_{file_number}.txt'

        # save the raw messages to a file
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(input_text)

        # save a reference to the raw messages in the metadata
        metadata['raw_messages'] = f'{self.raw_messages_folder}/{metadata["timestamp"]}.txt'

        # process each batch of sentences into triples and embeddings
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_text = '. '.join(batch)
            self._process_text_into_triples_and_embeddings(batch_text, metadata)

    @staticmethod
    def split_sentences(input_text):
        # split on newlines
        sentences = input_text.split('.')
        return sentences

    def _process_text_into_triples_and_embeddings(self, input_text, metadata=None, subject_threshold=0.9, verb_threshold=0.9,
                                                  object_threshold=0.9):
        # time the processing
        start_time = time.time()
        new_triples = self._extract_triples(input_text)
        end_time = time.time()
        print(f"Extracted {len(new_triples)} triples in {end_time - start_time} seconds")

        filtered_triples = []
        new_embeddings = self._triples_to_embeddings(new_triples)

        # Check each new triple for similarity to existing triples
        for new_emb, new_triple in zip(new_embeddings, new_triples):
            new_emb_flat = np.vstack(new_emb)  # Stack the embeddings into a 2D array
            faiss.normalize_L2(new_emb_flat)  # Normalize the embeddings if needed

            # Use the FAISS index to find the nearest existing embedding to the new embedding
            D, I = self.faiss_index.search(new_emb_flat, 1)  # search for the nearest neighbor

            # Adjust this condition to keep the triple if all parts are sufficiently dissimilar from existing triples
            if any(d > (1 - threshold) ** 2 for d, threshold in
                   zip(D, [subject_threshold, verb_threshold, object_threshold])):
                filtered_triples.append(new_triple)
                self.faiss_index.add(new_emb_flat)  # add the new embeddings to the FAISS index if they're unique
                print("Added triple: ", new_triple)

        print(f"Filtered {len(filtered_triples)} unique triples.")

        # Update the main triples list
        start_time = time.time()

        # When adding data to the FAISS index, also update the id_to_triple mapping:
        ids = np.arange(len(self.triples_list), len(self.triples_list) + len(filtered_triples))
        self.id_to_triple.update(dict(zip(ids, filtered_triples)))

        self.triples_list.extend(filtered_triples)

        # Add metadata to the metadata database
        if metadata is not None:
            metadata_entries = []  # Create a list to collect the metadata entries
            for triple_id, triple in zip(ids, filtered_triples):
                metadata_copy = metadata.copy()  # Create a copy of the metadata dictionary
                metadata_copy['triple_id'] = int(triple_id)
                metadata_entries.append(metadata_copy)  # Append the copy to the list

            self.metadata_db.insert_multiple(metadata_entries)  # Insert all metadata entries at once

        end_time = time.time()
        print(f"Added {len(filtered_triples)} triples to the list in {end_time - start_time} seconds")

    def _extract_triples(self, input_text):
        triples_list = []

        tuples_output = summarize_messages_tuples(self.client, input_text)

        # print("Got message: " + message.content)

        def clean_json(json_string):
            # Remove unescaped control characters
            json_string = re.sub(r'[\x00-\x1F]', '', json_string)
            return json_string

        # Parse the message.content and add to the graph
        data = json.loads(clean_json(tuples_output))
        for triple in data:
            try:
                subject = triple['subject']
                relationship = triple['relationship']
                obj = triple['object']
            except KeyError:
                print("Error: Triple missing subject, subject, or object")
                continue
            triples_list.append((subject, relationship, obj))

        return triples_list


    def summarize_graph(self, triples_list):
        # Convert the list of triples to a string
        input_text = str(triples_list)

        # Prepare the prompt for summarization
        summarize_triples_prompt = """
        You are tasked with very briefly summarizing the triples provided in the text provided. 
        Piece together the triples into a brief summary of the text while retaining as much information as possible.
        Also, summarize / cite any source information or other metadata.
        """

        # Combine the prompt with the input text
        summary_prompt = summarize_triples_prompt + "\n\n" + input_text

        summary_prompt += "\nSummary:"

        # Send the summarization prompt to the OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract the summary from the API response
        summary = response.choices[0].message.content.strip()

        return summary

    def _triples_to_embeddings(self, triples):
        subjects, verbs, objects = zip(*triples)  # Unzip the triples into separate lists
        all_texts = subjects + verbs + objects  # Concatenate all texts
        all_embeddings = self.embedding_model.encode(all_texts)  # Batch process all texts
        # Split the embeddings back into subjects, verbs, and objects
        subject_embeddings = all_embeddings[:len(subjects)]
        verb_embeddings = all_embeddings[len(subjects):len(subjects) + len(verbs)]
        object_embeddings = all_embeddings[len(subjects) + len(verbs):]
        svo_embeddings = list(zip(subject_embeddings, verb_embeddings, object_embeddings))
        return svo_embeddings

    def build_graph_from_noun(self, query, similarity_threshold=0.8, depth=0, metadata_query=None,
                              return_metadata=False):
        if len(self.triples_list) == 0:
            return
        if metadata_query is None:
            index = self.faiss_index
            id_mapping = IdentityMap()
            triples_list = self.triples_list
            if len(triples_list) == 0:
                return
        else:
            index, id_mapping, triples_list = self._filter_index_by_metadata_query(metadata_query)

        # Initialize lists to collect results and a set to keep track of visited nodes
        collected_triples = []
        collected_metadata = []
        visited = set()

        def recursive_search(current_point, current_depth):
            if current_depth > depth:
                return

            visited.add(current_point)
            current_point_embedding = self.embedding_model.encode([current_point])[0]

            # Query the FAISS index
            D, I = index.search(current_point_embedding.reshape(1, -1), len(triples_list) * 3)  # Adjusted the length

            for i in range(0, len(I[0])):
                idx = I[0][i]
                mapped_idx = id_mapping[idx // 3]    # Adjusted the mapping

                if mapped_idx in self.id_to_triple:
                    triple = self.id_to_triple[mapped_idx]
                    subject, _, object_ = triple

                    # Similarities are already computed in the FAISS index, so we can use them directly
                    similarity = 1 - D[0][i]

                    if similarity >= similarity_threshold:
                        collected_triples.append(triple)
                        if return_metadata:
                            Q = Query()
                            metadata_record = self.metadata_db.search(Q.triple_id == mapped_idx)
                            collected_metadata.append(metadata_record[0] if metadata_record else None)

                        # Recurse on the other side of the triple if it hasn't been visited yet
                        next_point = object_ if subject == current_point else subject
                        if next_point not in visited:
                            recursive_search(next_point, current_depth + 1)

        # Kick off the recursive search from the query point
        recursive_search(query, 0)

        if return_metadata:
            return list(zip(collected_triples, collected_metadata))
        else:
            return list(set(collected_triples))

    def visualize_graph_from_nouns(self, queries, similarity_threshold=0.8, depth=0, metadata_query=None):
        if len(self.triples_list) == 0:
            return

        if metadata_query is None:
            index = self.faiss_index
            id_mapping = IdentityMap()
            triples_list = self.triples_list
            if len(triples_list) == 0:
                return
        else:
            index, id_mapping, triples_list = self._filter_index_by_metadata_query(metadata_query)

        G = nx.DiGraph()
        visited = set()

        def recursive_search(current_point, current_depth):
            if current_depth > depth:
                return

            visited.add(current_point)
            current_point_embedding = self.embedding_model.encode([current_point])[0]
            D, I = index.search(current_point_embedding.reshape(1, -1), len(triples_list) * 3)

            for i in range(0, len(I[0])):
                idx = I[0][i]
                mapped_idx = id_mapping[idx // 3]

                if mapped_idx in self.id_to_triple:
                    triple = self.id_to_triple[mapped_idx]
                    subject, _, object_ = triple
                    similarity = 1 - D[0][i]

                    if similarity >= similarity_threshold:
                        G.add_edge(subject, object_, weight=similarity, label=f'Similarity: {similarity:.2f}')

                        next_point = object_ if subject == current_point else subject
                        if next_point not in visited:
                            recursive_search(next_point, current_depth + 1)

        for query in queries:
            recursive_search(query, 0)

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        edge_labels = {(node1, node2): data['label'] for node1, node2, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        nx.draw_networkx_labels(G, pos, font_size=12)
        plt.show()

    def build_graph_from_subject_relationship(self, subject_relationship, similarity_threshold=0.8, max_results=20, metadata_query=None,
                                      return_metadata=False):
        if len(self.triples_list) == 0:
            return

        if metadata_query is None:
            index = self.faiss_index
            id_mapping = IdentityMap()
        else:
            index, id_mapping, _ = self._filter_index_by_metadata_query(metadata_query)

        subject, verb = subject_relationship
        subject_embedding = self.embedding_model.encode([subject])[0]
        verb_embedding = self.embedding_model.encode([verb])[0]

        # Combine subject and verb embeddings into a single array
        query_embeddings = np.vstack([subject_embedding, verb_embedding])
        faiss.normalize_L2(query_embeddings)

        # Query the FAISS index
        D, I = index.search(query_embeddings, max_results)  # Search for the top 10 similar triples

        # Convert distances to similarities
        similarities = 1 - D

        # Collect triples that meet the similarity threshold for both subject and verb
        similar_triples = []
        similar_triples_metadata = []  # List to store metadata of similar triples

        # make a list of all elements common between I[0] // 3 and I[1] // 3
        S_indices = I[0] // 3
        V_indices = I[1] // 3
        common_elements = list(set(S_indices).intersection(V_indices))

        for i in range(len(common_elements)):
            idx = common_elements[i]  # Index of the similar triple
            s_index = np.argwhere(S_indices == idx)[0][0]
            v_index = np.argwhere(V_indices == idx)[0][0]

            subject_similarity = similarities[0][s_index]
            verb_similarity = similarities[1][v_index]
            if subject_similarity >= similarity_threshold and verb_similarity >= similarity_threshold:
                mapped_idx = id_mapping[idx]  # Map the index to the original index
                if mapped_idx in self.id_to_triple:
                    similar_triple = self.id_to_triple[mapped_idx]
                    similar_triples.append(similar_triple)
                    # Query metadata database for metadata of the similar triple
                    Q = Query()
                    metadata_record = self.metadata_db.search(Q.triple_id == mapped_idx)
                    if metadata_record:
                        similar_triples_metadata.append(
                            metadata_record[0])  # Assuming each triple_id has one metadata record
                    else:
                        similar_triples_metadata.append(None)  # Append None if no metadata found
                else:
                    print("Triple with ID {} not found in id_to_triple map".format(mapped_idx))

        if return_metadata:
            return list(zip(similar_triples, similar_triples_metadata))  # Return both triples and metadata
        else:
            return similar_triples  # Return only triples

    def _get_vector_by_id(self, triple_id, index=None):
        if index is None:
            index = self.faiss_index
        return index.reconstruct(triple_id)

    def _filter_index_by_metadata_query(self, metadata_query, index=None):
        if index is None:
            index = self.faiss_index

        # Get the IDs of the triples that match the metadata criteria
        matching_triple_ids = self._query_triple_ids(metadata_query)

        # Create a new temporary FAISS index
        d = self.embedding_dim  # assuming self.embedding_dim is the dimension of your embeddings
        temp_index = faiss.IndexFlatL2(d)

        # Collect the vectors corresponding to the matching triple IDs into a single numpy array
        # Adjusting the loop to account for the three consecutive indices per triple
        vectors = np.array([self._get_vector_by_id(triple_id * 3 + position)
                            for triple_id in matching_triple_ids
                            for position in range(3)])

        # Add the vectors to the temporary index in one batch
        temp_index.add(vectors)

        filtered_triples_list = [self.id_to_triple[triple_id] for triple_id in matching_triple_ids]

        # Adjusting the id_mapping to account for the three consecutive indices per triple
        id_mapping = {new_id: original_id for new_id, original_id in enumerate(matching_triple_ids)}

        return temp_index, id_mapping, filtered_triples_list

    def query_triples_from_metadata(self, metadata_criteria):
        matching_triple_ids = self._query_triple_ids(metadata_criteria)
        matching_triples = [self.id_to_triple[triple_id] for triple_id in matching_triple_ids]

        return matching_triples

    def _query_triple_ids(self, metadata_criteria):
        if not metadata_criteria:
            raise ValueError("metadata_criteria cannot be empty")
        Q = Query()
        # Construct the search condition dynamically from the metadata_criteria
        conditions = [getattr(Q, key) == value for key, value in metadata_criteria.items() if value is not None]
        if not conditions:
            raise ValueError("No valid conditions provided in metadata_criteria")
        search_condition = reduce(lambda a, b: a & b, conditions)
        matching_records = self.metadata_db.search(search_condition)
        matching_triple_ids = [record['triple_id'] for record in matching_records]
        return matching_triple_ids

    def get_summaries_from_topics(self, topics, similarity_threshold=0.5):
        if not topics:
            return ""
        #if no triples exist, return empty string
        if len(self.triples_list) == 0:
            return ""

        summaries_with_headings = ["The following is what you know about the current topics:\n\n"]
        print(f"Getting summaries for topics: {topics}")
        for topic in topics:
            print(f"Building graph summary for topic: {topic}")
            triples = self.build_graph_from_noun(topic, similarity_threshold=similarity_threshold)
            if len(triples) > 0:
                print(f"Found {len(triples)} triples for topic: {topic}")
                print(triples)
                summary = self.summarize_graph(triples)
                summaries_with_headings.append(f"{topic}:\n{summary}\n")
                print(f"Summary for topic {topic}:\n{summary}")

        if len(summaries_with_headings) == 0:
            return ""
        else:
            return "\n".join(summaries_with_headings)


# run to test
if __name__ == '__main__':

    # turn on tests as needed
    graph_test = False
    visualize_test = False
    recall_test = True

    if graph_test:
        load_secrets()
        client = OpenAI(
            api_key="sk-111111111111111111111111111111111111111111111111",
            base_url=os.environ['LOCAL_TEXTGEN_API_BASE']
        )

        # delete the graph store memory
        if os.path.exists("Test_GraphStoreMemory"):
            shutil.rmtree("Test_GraphStoreMemory")

        # Create a vector knowledge graph
        kgraph = VectorKnowledgeGraph(chat_llm=client, path="Test_GraphStoreMemory")

        # try to load the graph from file
        loaded = kgraph.load()

        if not loaded:
            # Example usage
            text_1 = ("""Rachel is a young vampire girl with pale skin, long blond hair tied into two pigtails with black 
            ribbons, and red eyes. She wears Gothic Lolita fashion with a frilly black gown and jacket, red ribbon bow tie, 
            a red bat symbol design cross from the front to the back on her dress, another red cross on her shawl and bottom 
            half, black pointy heel boots with a red cross, and a red ribbon on her right ankle. Physically, Rachel is said 
            to look around 12 years old, however, she gives off an aura of someone far older than what she looks. 
        
            When she was young, her appearance was similar that of her current self. She wore a black dress with a red 
            cross in the center and a large, black ribbon on the back, black ribbons in her hair, a white blouse, 
            white bloomers, and black slippers. 
            
            In BlazBlue: Alter Memory during the hot spring scene in Episode 5, Rachel is seen wearing a dark blue one-piece 
            bathing suit with red lines on both sides. 
            
            Rachel bears an incredibly striking resemblance to Raquel Alucard, save that the two have a very different dress 
            sense, have different eye colors, hair style and a difference in appearance of age. """,
                    "https://blazblue.fandom.com/wiki/Rachel_Alucard#Appearance")

            text_2 = ("""Rachel is a stereotypical aristocratic heiress. She has an almost enchanting air of dignity and grace, 
            yet is sarcastic and condescending to those she considers lower than her, always expecting them to have the 
            highest standards of formality when conversing with her. Despite this, she does care deeply for her allies. Her 
            butler, Valkenhayn, is fervently devoted to Rachel, as he was a loyal friend and respected rival to her father, 
            the late Clavis Alucard, and she, in turn, treats him with a greater level of respect than anyone else. Rachel’s 
            two familiars, Nago and Gii, despite taking punishment from her on a regular basis, remain loyal to her. Perhaps 
            her most intriguing relationship is with Ragna. Although Rachel would never admit to it, she loves Ragna for his 
            determination and unwillingness to give up even when the odds are against him, wanting him to reach his full 
            potential as a warrior and as a person. In BlazBlue: Centralfiction, her feelings for Ragna become more evident 
            as revealed in her arcade mode. She becomes even more concerned when she finds out that Naoto’s existence is 
            affecting Ragna. This is most notably the only time she lost her composure. In the end of the game, Rachel sheds 
            a tear over his large sword, despite forgetting Ragna. ""","https://blazblue.fandom.com/wiki/Rachel_Alucard#Personality")

            text_3 = ("""Ragna is sardonic, rude, and abrasive to anyone he comes 
            across. He is also quick to anger, stubborn, and never misses a chance to use as much vulgar language as 
            possible. In this regard, Ragna is similar to the stereotypical anime delinquent. This is caused mainly by Yūki 
            Terumi practically destroying Ragna’s life, which has created a mass of hatred in him; stronger than that of any 
            other individual. Ragna often becomes infuriated at first sight of Yūki Terumi, which he and/or Hazama often 
            takes advantage of through taunting him. However, even in cases where he cannot win or is on the brink of death, 
            Ragna possesses an undying will and persistence, refusing to give up even when he is clearly out matched, 
            something many characters either hate or admire. 
        
            Beneath his gruff exterior, however, Ragna does possess a softer, more compassionate side. He chooses to keep up his 
            public front because of the path he chose – that of revenge, as well as accepting the fact that he’s still someone 
            who’s committed crimes such as murder. He does genuinely care for certain people, such as Taokaka, Rachel, Noel, 
            Jūbei and, to an extent, his brother, Jin""", "https://blazblue.fandom.com/wiki/Ragna_the_Bloodedge#Personality")

            # load sample log to string
            # with open('Sophia_logs/2023-09-09.txt', 'r') as file:
            #    text = file.read().replace('\n', '')

            print("Processing text...")
            start = time.time()
            kgraph.process_text(text_1[0], {"reference": text_1[1]})
            kgraph.process_text(text_2[0], {"reference": text_2[1]})
            kgraph.process_text(text_3[0], {"reference": text_3[1]})
            print(time.time() - start)

            kgraph.save()

        text_triples = kgraph.triples_list

        print("Text triples:")
        print(text_triples)

        print("Building graph...")
        start = time.time()
        query = "Rachel"
        graph = kgraph.build_graph_from_noun(query, 0.7, 0)
        print(time.time() - start)
        print("Query: " + query)
        print(graph)

        print("Building graph...")
        start = time.time()
        query = "Ragna"
        graph = kgraph.build_graph_from_noun(query, 0.7, 0)
        print(time.time() - start)
        print("Query: " + query)
        print(graph)
        print("Summarizing graph...")
        print(kgraph.summarize_graph(graph))

        print("Building graph using filter...")
        start = time.time()
        query = "Ragna"
        metadata_filter = {"reference": "https://blazblue.fandom.com/wiki/Ragna_the_Bloodedge#Personality"}
        graph = kgraph.build_graph_from_noun(query, 0.7, 0, metadata_query=metadata_filter, return_metadata=True)
        print(time.time() - start)
        print("Query: " + query)
        print(graph)

        print("Building graph...")
        start = time.time()
        query = "Rachel"
        graph = kgraph.build_graph_from_noun(query, 0.7, 1)
        print(time.time() - start)
        print("Query: " + query)
        print(graph)

        print("Building graph...")
        start = time.time()
        subject_relationship_tuple = ('Rachel', 'wears')

        graph = kgraph.build_graph_from_subject_relationship(subject_relationship_tuple, similarity_threshold=0.7, return_metadata=True)
        print(time.time() - start)
        print(subject_relationship_tuple)
        print(graph)
        print("Summary: ")
        print(kgraph.summarize_graph(graph))

        print("getting all items from Ragna Article")
        start = time.time()
        triples = kgraph.query_triples_from_metadata({"reference": "https://blazblue.fandom.com/wiki/Ragna_the_Bloodedge#Personality"})
        print(time.time() - start)
        print(triples)

    if visualize_test:
        load_secrets()
        k_graph = VectorKnowledgeGraph(path="Test_GraphStoreMemory")
        k_graph.visualize_graph_from_nouns(["Rachel"], similarity_threshold=0.6, depth=3)

    if recall_test:
        from AIAgent import AIAgent
        load_secrets()

        # delete the graph store memory
        if os.path.exists("Test_GraphStoreMemory"):
            shutil.rmtree("Test_GraphStoreMemory")

        k_graph = VectorKnowledgeGraph(path="Test_GraphStoreMemory")

        # delete the graph store memory
        if os.path.exists("TestAgent_logs"):
            shutil.rmtree("TestAgent_logs")

        # create a profile map for the agent
        profile_map = {
            "name": "TestAgent",
            "voice": "None",
            "personality": "I am an AI agent designed to assist you with your queries."
        }

        test_agent = AIAgent("TestAgent", profile_map)

        counter = 0

        while True:
            # prompt user for input
            in_message = input("Enter a message: ")
            # send message to agent
            test_response, test_emotion, test_monologue, test_actions = test_agent.send(in_message)
            # print response
            print(test_response)

            # every 5 interactions, save last 10 messages
            if counter % 5 == 0 and counter != 0:
                messages = test_agent.messages[-10:]
                k_graph.process_text(messages, {"reference": "TestAgent_logs"})

            counter += 1


