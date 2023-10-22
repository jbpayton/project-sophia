import json
import time
import faiss
import os

from collections import deque, defaultdict

import numpy as np

from util import load_secrets

from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.schema import SystemMessage, HumanMessage


class EmbeddingKnowledgeGraph:
    def __init__(self, chat_llm=None, embedding_model=None, embedding_dim=384):
        if chat_llm is None:
            self.chat_llm = ChatOpenAI(
                model_name='gpt-4',
                temperature=0.0
            )
        else:
            self.chat_llm = chat_llm

        if embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            self.embedding_model = embedding_model
            self.embedding_dim = embedding_dim

        self.triples_list = []
        self.id_to_triple = {}
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def save(self, path="EmbeddingKnowledgeGraph"):
        try:
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

    def load(self, path="EmbeddingKnowledgeGraph"):
        try:
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

    def process_text(self, input_text, batch_size=50):
        # separate the text into sentences
        sentences = self.split_sentences(input_text)

        # process each batch of sentences into triples and embeddings
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_text = '. '.join(batch)
            self.process_text_into_triples_and_embeddings(batch_text)

    def split_sentences(self, input_text):
        sentences = input_text.split('.')
        return sentences

    def process_text_into_triples_and_embeddings(self, input_text, subject_threshold=0.9, verb_threshold=0.9,
                                                 object_threshold=0.9):
        # time the processing
        start_time = time.time()
        new_triples = self.extract_triples(input_text)
        end_time = time.time()
        print(f"Extracted {len(new_triples)} triples in {end_time - start_time} seconds")

        filtered_triples = []
        new_embeddings = self.triples_to_embeddings(new_triples)

        # Set up a FAISS index if it doesn't exist
        if not hasattr(self, 'faiss_index'):
            d = new_embeddings[0][0].shape[0]  # dimension of embeddings
            self.faiss_index = faiss.IndexFlatL2(d)  # use a flat L2 index for simplicity

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

        print(f"Filtered {len(filtered_triples)} unique triples.")

        # Update the main triples list
        start_time = time.time()

        # When adding data to the FAISS index, also update the id_to_triple mapping:
        ids = np.arange(len(self.triples_list), len(self.triples_list) + len(filtered_triples))
        self.id_to_triple.update(dict(zip(ids, filtered_triples)))

        self.triples_list.extend(filtered_triples)

        end_time = time.time()
        print(f"Added {len(filtered_triples)} triples to the list in {end_time - start_time} seconds")

    def extract_triples(self, input_text):
        triples_list = []
        build_topic_graphs_prompt = f"""
        You are tasked with extracting factual information from the text provided in the form of (subject, predicate, object) triples. 
        Adhere to the following guidelines to ensure accuracy and granularity:
        1. Extract fundamental relationships, ignoring conversational fillers, greetings, and sign-offs.
        2. Decompose complex triples into simpler, more atomic triples. Each triple should represent a single, clear relationship or attribute.
        3. Make the object of each triple as singular or atomic as possible.
        4. Use a consistent verb form for predicates, preferably base form (e.g., "love" instead of "loves", "work" instead of "working").
        5. Capture specific relationships and attributes rather than generic conversational phrases.
        6. Choose verbs that accurately convey the core meaning of the action or relationship. For instance, use 'cries' instead of 'sheds a tear'.
        7. Do not smush multiple words into one. For example, "wants to learn about" should not be "wantsToLearnAbout".
        8. Make items in triples as compact as possible without losing meaning.
        9. Do all of the above from a summarization of the text, not from the text itself.
        Format the output as JSON like this: 
        {{ "triples": [{{"subject": "entity 1", "predicate": "predicate", "object": "entity 2"}}, ...] }}
        """

        message = self.chat_llm(
            [
                SystemMessage(role="TripleExtractor", content=build_topic_graphs_prompt),
                HumanMessage(content=input_text),
            ]
        )

        # print("Got message: " + message.content)

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
            triples_list.append((subject, predicate, obj))

        return triples_list

    def triples_to_embeddings(self, triples):
        subjects, verbs, objects = zip(*triples)  # Unzip the triples into separate lists
        all_texts = subjects + verbs + objects  # Concatenate all texts
        all_embeddings = self.embedding_model.encode(all_texts)  # Batch process all texts
        # Split the embeddings back into subjects, verbs, and objects
        subject_embeddings = all_embeddings[:len(subjects)]
        verb_embeddings = all_embeddings[len(subjects):len(subjects) + len(verbs)]
        object_embeddings = all_embeddings[len(subjects) + len(verbs):]
        svo_embeddings = list(zip(subject_embeddings, verb_embeddings, object_embeddings))
        return svo_embeddings

    def get_most_similar_point(self, target_point, target_point_embedding, similarity_threshold):
        # Reshape the target_point_embedding to match the expected input shape for faiss
        query_embeddings = target_point_embedding.reshape(1, -1)

        # Query the FAISS index
        D, I = self.faiss_index.search(query_embeddings, 1)

        # Get the ID and distance of the most similar point
        most_similar_point_id = I[0][0]

        #if point is not in the id to triple map (i.e., the same as target point), return None
        if most_similar_point_id not in self.id_to_triple:
            return None

        distance = D[0][0]

        # Convert the distance to similarity
        similarity_score = 1 - distance

        # Check if the similarity_score meets the similarity_threshold
        if similarity_score >= similarity_threshold:
            # Retrieve the triple associated with the most_similar_point_id from the id_to_triple map
            most_similar_triple = self.id_to_triple[most_similar_point_id]
            # Extract the subject or object of the triple (whichever is not the target_point)
            most_similar_point = most_similar_triple[0] if most_similar_triple[0] != target_point else \
            most_similar_triple[2]
            return most_similar_point

        return None  # Return None if no point meets the similarity_threshold

    def build_graph_from_noun(self, query, similarity_threshold=0.5, depth=0):
        svo_text = self.triples_list
        collected_triples = []
        visited = set()
        svo_index = defaultdict(list)
        for svo_txt in svo_text:
            subject, _, object_ = svo_txt
            svo_index[subject].append(svo_txt)
            svo_index[object_].append(svo_txt)

        # Obtain the embedding for the query
        query_embedding = self.embedding_model.encode([query])[0]

        queue = deque(
            [(query, True, 0), (query, False, 0)])  # Initialize queue with subject and object view of query at depth 0

        while queue:
            current_point, is_subject, current_depth = queue.popleft()
            if current_depth > depth:  # Stop processing once depth exceeds the specified depth
                continue
            visited.add(current_point)

            # If the current point is not the initial query, obtain its embedding
            if current_point != query:
                current_point_embedding = self.embedding_model.encode([current_point])[0]
            else:
                current_point_embedding = query_embedding

            for svo_txt in svo_index[current_point]:
                subject, verb, object_ = svo_txt
                if (is_subject and subject == current_point) or (not is_subject and object_ == current_point):
                    collected_triples.append(svo_txt)
                    next_point = object_ if subject == current_point else subject
                    next_is_subject = subject == current_point
                    if next_point not in visited:
                        queue.append((next_point, next_is_subject, current_depth + 1))  # Increment depth for each step

            # Look for most similar point both as a subject and as an object
            for is_subject in [True, False]:
                most_similar_point = self.get_most_similar_point(
                    current_point,
                    current_point_embedding,
                    similarity_threshold
                )
                if most_similar_point:
                    queue.append((most_similar_point, is_subject, current_depth + 1))  # Increment depth for each step

        collected_triples = list(set(collected_triples))
        return collected_triples

    def build_graph_from_subject_verb(self, subject_verb, similarity_threshold=0.6, max_results=10):
        subject, verb = subject_verb
        subject_embedding = self.embedding_model.encode([subject])[0]
        verb_embedding = self.embedding_model.encode([verb])[0]

        # Combine subject and verb embeddings into a single array
        query_embeddings = np.vstack([subject_embedding, verb_embedding])
        faiss.normalize_L2(query_embeddings)

        # Query the FAISS index
        D, I = self.faiss_index.search(query_embeddings, max_results)  # Search for the top 10 similar triples

        # Convert distances to similarities
        similarities = 1 - D

        # Collect triples that meet the similarity threshold for both subject and verb
        similar_triples = []
        for i in range(len(I[0])):
            idx = I[0][i]  # Index of the similar triple
            subject_similarity = similarities[0][i]
            verb_similarity = similarities[1][i]
            if subject_similarity >= similarity_threshold and verb_similarity >= similarity_threshold:
                similar_triple = self.id_to_triple[idx]
                similar_triples.append(similar_triple)

        return similar_triples


# run to test
if __name__ == '__main__':
    load_secrets()

    kgraph = EmbeddingKnowledgeGraph()

    #try to load the graph from file
    loaded = kgraph.load()

    if not loaded:
        # Example usage
        text = """Rachel is a young vampire girl with pale skin, long blond hair tied into two pigtails with black 
        ribbons, and red eyes. She wears Gothic Lolita fashion with a frilly black gown and jacket, red ribbon bow tie, 
        a red bat symbol design cross from the front to the back on her dress, another red cross on her shawl and bottom 
        half, black pony heel boots with a red cross, and a red ribbon on her right ankle. Physically, Rachel is said to 
        look around 12 years old, however, she gives off an aura of someone far older than what she looks. 
    
        When she was young, her appearance was similar that of her current self. She wore a black dress with a red cross in 
        the center and a large, black ribbon on the back, black ribbons in her hair, a white blouse, white bloomers, 
        and black slippers. 
        
        In BlazBlue: Alter Memory during the hot spring scene in Episode 5, Rachel is seen wearing a dark blue one-piece 
        bathing suit with red lines on both sides. 
        
        Rachel bears an incredibly striking resemblance to Raquel Alucard, save that the two have a very different dress 
        sense, have different eye colors, hair style and a difference in appearance of age. 
    
        Rachel is a stereotypical aristocratic heiress. She has an almost enchanting air of dignity and grace, 
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
        a tear over his large sword, despite forgetting Ragna. Ragna is sardonic, rude, and abrasive to anyone he comes 
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
        Jūbei and, to an extent, his brother, """

        # load sample log to string
        #with open('Sophia_logs/2023-09-09.txt', 'r') as file:
        #    text = file.read().replace('\n', '')

        print("Processing text...")
        start = time.time()
        kgraph.process_text(text)
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

    print("Building graph...")
    start = time.time()
    query = "Rachel"
    graph = kgraph.build_graph_from_noun(query, 0.7, 1)
    print(time.time() - start)
    print("Query: " + query)
    print(graph)

    print("Building graph...")
    start = time.time()
    subject_verb_tuple = ('Rachel', 'wears')

    graph = kgraph.build_graph_from_subject_verb(subject_verb_tuple, similarity_threshold=0.8)
    print(time.time() - start)
    print(subject_verb_tuple)
    print(graph)
