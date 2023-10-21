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
                model_name='gpt-3.5-turbo'
            )
        else:
            self.chat_llm = chat_llm

        if embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedding_model = embedding_model

        self.triples_list = []
        self.embeddings_list = []
        self.similarity_cache = {}

    def process_text(self, input_text, batch_size=10):
        # separate the text into sentences
        sentences = self.split_sentences(input_text)

        # process each batch of sentences into triples and embeddings
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_text = '. '.join(batch)
            self.process_text_into_triples_and_embeddings(batch_text)

    def split_sentences(self, input_text):
        sentences = input_text.split('.')
        return sentences

    def process_text_into_triples_and_embeddings(self, input_text):
        triples = self.extract_triples(input_text)
        self.triples_list.extend(triples)
        embeddings = self.triples_to_embeddings(triples)
        self.embeddings_list.extend(embeddings)

    def extract_triples(self, input_text):
        triples_list = []
        build_topic_graphs_prompt = f"""You are an AI who reads conversations and builds RDF 
        triples of the form (subject, predicate, object), e.g., (I, like, apples) or (Bob, is a father to, Ann).
    
        Consider the following guidelines while 
        identifying the RDF triples: 
    
        1. **Verb Form**: Use consistent verb forms (preferably base form) for predicates, e.g., "support", "love", 
        "dislike". 2. **Specific Relationships**: Identify specific, directional verbs that clearly indicate the nature 
        of the relationship between the subject and the object, e.g., "is a parent of", "works at". 3. Make sure all triples are logically consistent
    
        Please proceed with generating the knowledge graph based on the conversation provided.
        """

        request_prompt = """\nPlease get a list of RDF triples (subject, predicate, object), from this conversation.
                         Format this in in JSON, in this format:
                         {
                           \"triples\": [\r\n
                             {\r\n
                               \"subject\": \"entity 1\",\r\n
                               \"predicate\": \"predicate\",\r\n
                               \"object\": \"entity 2\"\r\n
                             },...
                          ],\r\n
                         \nTriples should link attributes and relationships to the subject,
                         rather than simply stating what is said:\n"""

        message = self.chat_llm(
            [
                SystemMessage(role="TripleExtractor", content=build_topic_graphs_prompt),
                HumanMessage(content=request_prompt + input_text),
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

    def build_graph_from_noun_4(self, starting_point, similarity_threshold, depth):
        svo_embeddings = self.embeddings_list
        svo_text = self.triples_list
        collected_triples = []
        visited = set()
        svo_index = defaultdict(list)
        for svo_emb, svo_txt in zip(svo_embeddings, svo_text):
            subject, _, object_ = svo_txt
            svo_index[subject].append((svo_emb, svo_txt))
            svo_index[object_].append((svo_emb, svo_txt))

        def get_most_similar_point(target_point, target_is_subject, candidate_triples):
            max_similarity = 0
            most_similar_point = None
            for candidate_emb, candidate_txt in candidate_triples:
                candidate_point = candidate_txt[0] if target_is_subject else candidate_txt[2]
                if candidate_point and candidate_point not in visited:
                    cache_key = (current_point, candidate_point)
                    if cache_key in self.similarity_cache:
                        similarity = self.similarity_cache[cache_key]
                    else:
                        similarity = cosine_similarity(
                            target_point.reshape(1, -1),
                            candidate_emb[0 if target_is_subject else 2].reshape(1, -1)
                        )[0][0]
                        self.similarity_cache[cache_key] = similarity
                    if similarity > max_similarity and similarity >= similarity_threshold:
                        max_similarity = similarity
                        most_similar_point = candidate_point
                    if similarity > 0.99:
                        break
            return most_similar_point

        queue = deque([(starting_point, True, 0), (starting_point, False, 0)])

        while queue:
            current_point, is_subject, current_depth = queue.popleft()
            if current_depth == depth:
                continue
            visited.add(current_point)
            for svo_emb, svo_txt in svo_index[current_point]:
                subject, verb, object_ = svo_txt
                if (is_subject and subject == current_point) or (not is_subject and object_ == current_point):
                    collected_triples.append(svo_txt)
                    next_point = object_ if subject == current_point else subject
                    next_is_subject = subject == current_point
                    if next_point not in visited:
                        queue.append((next_point, next_is_subject, current_depth + 1))
            most_similar_point = get_most_similar_point(
                svo_embeddings[0][0 if is_subject else 2],
                is_subject,
                svo_index[current_point]
            )
            if most_similar_point:
                queue.append((most_similar_point, is_subject, current_depth + 1))

        collected_triples = list(set(collected_triples))
        return collected_triples


# run to test
if __name__ == '__main__':
    load_secrets()

    kgraph = EmbeddingKnowledgeGraph()

    # Example usage
    text = """Rachel is a stereotypical aristocratic heiress. She has an almost enchanting air of dignity and grace, 
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

    kgraph.process_text(text)
    text_triples = kgraph.triples_list
    embedding_triples = kgraph.embeddings_list
    print(text_triples)

    start = time.time()
    graph = kgraph.build_graph_from_noun_4("Ragna", 0.80, 1)
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
    object_1_to_subject_3_similarity = cosine_similarity(object_embedding_1.reshape(1, -1), subject_embedding_3.reshape(1, -1))[0][0]
    verb_similarity = cosine_similarity(verb_embedding_2.reshape(1, -1), verb_embedding_3.reshape(1, -1))[0][0]
    object_similarity = cosine_similarity(object_embedding_2.reshape(1, -1), object_embedding_3.reshape(1, -1))[0][0]

    # Print out the similarity
    print(f"Cosine similarity between object_1 and subject_3: {object_1_to_subject_3_similarity:.2f}")
    print(f"Cosine similarity between verbs 2 and 3: {verb_similarity:.2f}")
    print(f"Cosine similarity between objects 2 and 3: {object_similarity:.2f}")