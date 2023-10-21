import json
import time
import torch
from collections import deque, defaultdict

from util import load_secrets

from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import SystemMessage, HumanMessage


class EmbeddingKnowledgeGraph:

    def __init__(self, chat_llm=None, embedding_model=None):
        if chat_llm is None:
            self.chat_llm = ChatOpenAI(
                model_name='gpt-3.5-turbo',
                temperature=0.0
            )
        else:
            self.chat_llm = chat_llm

        if embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedding_model = embedding_model

        self.triples_list = []
        self.embedding_cache = {} # Key: triple, Value: (subject_embedding, verb_embedding, object_embedding)
        self.similarity_cache = {} # Key: (triple1, triple2), Value: (subject_similarity, verb_similarity, object_similarity)

    def process_text(self, input_text, batch_size=20):
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

    def triple_similarity(self, triple1, triple2):
        similarities = []
        for emb1, emb2 in zip(triple1, triple2):
            similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
            similarities.append(similarity)
        return similarities

    def process_text_into_triples_and_embeddings(self, input_text, subject_threshold=0.70, verb_threshold=0.6,
                                                 object_threshold=0.6):
        # time the processing
        start_time = time.time()
        new_triples = self.extract_triples(input_text)
        end_time = time.time()
        print(f"Extracted {len(new_triples)} triples in {end_time - start_time} seconds")

        filtered_triples = []
        filtered_embeddings = []
        new_embeddings = self.triples_to_embeddings(new_triples)

        # Filter new triples based on similarity
        for new_emb, new_triple in zip(new_embeddings, new_triples):
            is_similar = False
            for existing_triple, existing_emb in self.embedding_cache.items():
                pair = (new_triple, existing_triple)
                if pair not in self.similarity_cache:
                    # Compute similarity and store in the cache
                    similarities = self.triple_similarity(new_emb, existing_emb)
                    self.similarity_cache[pair] = similarities
                else:
                    # Retrieve similarity from the cache
                    similarities = self.similarity_cache[pair]
                # Check if similarities are above the thresholds
                if (similarities[0] >= subject_threshold and
                        similarities[1] >= verb_threshold and
                        similarities[2] >= object_threshold):
                    is_similar = True
                    break  # Break if any existing triple is too similar to the new triple
            if not is_similar:
                filtered_triples.append(new_triple)
                filtered_embeddings.append(new_emb)

        print(f"Filtered {len(filtered_triples)} unique triples.")

        # Update the main embedding cache and triples list
        start_time = time.time()
        self.triples_list.extend(filtered_triples)
        self.embedding_cache.update({triple: emb for triple, emb in zip(filtered_triples, filtered_embeddings)})
        end_time = time.time()
        print(f"Added {len(filtered_triples)} triples to the list in {end_time - start_time} seconds")

    def extract_triples(self, input_text):
        triples_list = []
        build_topic_graphs_prompt = f""" You are an AI tasked with extracting factual information from a series of 
        text-based conversations. Your goal is to identify relationships between entities and actions or attributes 
        pertaining to those entities. You should focus on extracting triples of the form (subject, predicate, 
        object), like (Joey, wants to learn about, bats) or (Sophia, is excited about, learning). 

            To ensure clarity and granularity in the information you extract, adhere to the following guidelines:
            1. Decompose complex triples into a 'subgraph' of simpler ones, whenever possible. Each triple should represent a single, clear relationship or attribute.
            2. Ignore conversational fillers, greetings, and sign-offs.
            3. Focus on the core informational content in each statement.
            4. Make the object of each triple as singular or atomic as possible.
            5. Use a consistent verb form for predicates, preferably base form (e.g., "love" instead of "loves", "work" instead of "working").
            6. Capture specific relationships and attributes rather than generic conversational phrases.
            7. When a person asks a question, try to infer the underlying statement or desire rather than just capturing the act of asking.
            8. Do not smush multiple words into one. For example, "wants to learn about" should not be "wantsToLearnAbout".
            9. Make items in triples as compact as possible without losing meaning.
        """

        request_prompt = """Please proceed with generating a list of subject-predicate-object triples from the text 
        provided. Your output should be formatted as JSON, with each triple containing a subject, predicate, 
        and object. 

        Now, please extract the factual information from the following conversation and present it in the specified 
        format: Format this in in JSON, in this format: { "triples": [\r\n {\r\n "subject": "entity 1",
        \r\n "predicate": "predicate", \r\n "object": "entity 2"\r\n },... ],\r\n \nTriples should link attributes 
        and relationships to the subject, rather than simply stating what is said:\n """

        message = self.chat_llm(
            [
                SystemMessage(role="TripleExtractor", content=build_topic_graphs_prompt),
                HumanMessage(content=request_prompt + input_text),
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

    def build_graph_from_noun(self, query, similarity_threshold, depth):
        svo_text = self.triples_list
        collected_triples = []
        visited = set()
        svo_index = defaultdict(list)
        for svo_txt in svo_text:
            subject, _, object_ = svo_txt
            svo_emb = self.embedding_cache[svo_txt]
            svo_index[subject].append((svo_emb, svo_txt))
            svo_index[object_].append((svo_emb, svo_txt))

        def get_most_similar_point(target_point, target_is_subject):
            max_similarity = 0
            most_similar_point = None
            for svo_emb, svo_txt in all_triples:  # Iterate over all triples
                candidate_point = svo_txt[0] if target_is_subject else svo_txt[2]
                if candidate_point and candidate_point not in visited:
                    similarity = cosine_similarity(
                        target_point.reshape(1, -1),
                        svo_emb[0 if target_is_subject else 2].reshape(1, -1)
                    )[0][0]
                    if similarity > max_similarity and similarity >= similarity_threshold:
                        max_similarity = similarity
                        most_similar_point = candidate_point
                    if similarity > 0.99:
                        break
            return most_similar_point

        all_triples = []  # A list to hold all your triples
        for svo_txt in svo_text:
            subject, _, object_ = svo_txt
            svo_emb = self.embedding_cache[svo_txt]
            all_triples.append((svo_emb, svo_txt))

        queue = deque([(query, True, 0), (query, False, 0)])

        while queue:
            current_point, is_subject, current_depth = queue.popleft()
            if current_depth == depth:
                continue
            visited.add(current_point)

            current_point_embedding = None
            for triple, embedding in self.embedding_cache.items():
                if triple[0] == current_point or triple[2] == current_point:
                    current_point_embedding = embedding[0] if triple[0] == current_point else embedding[2]
                    break  # Exit the loop once the embedding is found

            if current_point_embedding is None:
                # Generate a new embedding for the query if it's not already in the cache
                current_point_embedding = self.embedding_model.encode([current_point])[0]

            for svo_emb, svo_txt in svo_index[current_point]:
                subject, verb, object_ = svo_txt
                if (is_subject and subject == current_point) or (not is_subject and object_ == current_point):
                    collected_triples.append(svo_txt)
                    next_point = object_ if subject == current_point else subject
                    next_is_subject = subject == current_point
                    if next_point not in visited:
                        queue.append((next_point, next_is_subject, current_depth + 1))

            most_similar_point = get_most_similar_point(
                current_point_embedding,
                is_subject
            )  # Note: no longer passing in candidate_triples

            if most_similar_point:
                queue.append((most_similar_point, is_subject, current_depth + 1))

        collected_triples = list(set(collected_triples))
        return collected_triples


# run to test
if __name__ == '__main__':
    load_secrets()

    kgraph = EmbeddingKnowledgeGraph()

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
    yet is sarcastic and condescending to those she considers lower than her, always expecting them to have the highest 
    standards of formality when conversing with her. Despite this, she does care deeply for her allies. Her butler, 
    Valkenhayn, is fervently devoted to Rachel, as he was a loyal friend and respected rival to her father, 
    the late Clavis Alucard, and she, in turn, treats him with a greater level of respect than anyone else. Rachel’s two 
    familiars, Nago and Gii, despite taking punishment from her on a regular basis, remain loyal to her. Perhaps her most 
    intriguing relationship is with Ragna. Although Rachel would never admit to it, she loves Ragna for his determination 
    and unwillingness to give up even when the odds are against him, wanting him to reach his full potential as a warrior 
    and as a person. In BlazBlue: Centralfiction, her feelings for Ragna become more evident as revealed in her arcade 
    mode. She becomes even more concerned when she finds out that Naoto’s existence is affecting Ragna. This is most 
    notably the only time she lost her composure. In the end of the game, Rachel sheds a tear over his large sword, 
    despite forgetting Ragna. """

    # load sample log to string
    with open('Sophia_logs/2023-09-09.txt', 'r') as file:
        text = file.read().replace('\n', '')

    print("Processing text...")
    start = time.time()
    kgraph.process_text(text)
    print(time.time() - start)

    text_triples = kgraph.triples_list
    # get embedding triples from embedding cache
    embedding_triples = [kgraph.embedding_cache[triple] for triple in text_triples]



    print(text_triples)

    print("Building graph...")
    start = time.time()
    graph = kgraph.build_graph_from_noun("graphics", 0.5, 2)
    print(time.time() - start)
    print(graph)

    # Extract embeddings of the first and second SVO triples
    subject_embedding_1, verb_embedding_1, object_embedding_1 = embedding_triples[0]
    subject_embedding_2, verb_embedding_2, object_embedding_2 = embedding_triples[1]

    # Compute cosine similarity between corresponding parts
    subject_similarity = cosine_similarity(subject_embedding_1.reshape(1, -1), subject_embedding_2.reshape(1, -1))[0][0]
    verb_similarity = cosine_similarity(verb_embedding_1.reshape(1, -1), verb_embedding_2.reshape(1, -1))[0][0]
    object_similarity = cosine_similarity(object_embedding_1.reshape(1, -1), object_embedding_2.reshape(1, -1))[0][0]

    # Print out the similarity
    print(f"Cosine similarity between subjects 1 and 2: {subject_similarity:.2f}")
    print(f"Cosine similarity between verbs 1 and 2: {verb_similarity:.2f}")
    print(f"Cosine similarity between objects 1 and 2: {object_similarity:.2f}")

    subject_embedding_3, verb_embedding_3, object_embedding_3 = embedding_triples[2]

    # Compute cosine similarity between corresponding parts
    object_1_to_subject_3_similarity = \
    cosine_similarity(object_embedding_1.reshape(1, -1), subject_embedding_3.reshape(1, -1))[0][0]
    verb_similarity = cosine_similarity(verb_embedding_2.reshape(1, -1), verb_embedding_3.reshape(1, -1))[0][0]
    object_similarity = cosine_similarity(object_embedding_2.reshape(1, -1), object_embedding_3.reshape(1, -1))[0][0]

    # Print out the similarity
    print(f"Cosine similarity between object_1 and subject_3: {object_1_to_subject_3_similarity:.2f}")
    print(f"Cosine similarity between verbs 2 and 3: {verb_similarity:.2f}")
    print(f"Cosine similarity between objects 2 and 3: {object_similarity:.2f}")
