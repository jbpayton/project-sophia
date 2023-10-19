import json
from util import load_secrets

from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import SystemMessage, HumanMessage

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

load_secrets()
chat_llm = ChatOpenAI(
            model_name='gpt-4'
        )


def extract_triples(input_text):
    triples_list = []
    build_topic_graphs_prompt = f"""You are an AI who reads conversations and builds RDF 
    triples of the form (subject, predicate, object), e.g., (I, like, apples) or (Bob, is a father to, Ann). If a relationship can be
    inverted, then capture the inverse as well, e.g., (Bob, likes, Ann) should also capture (Ann, is liked by, Bob).

    Consider the following guidelines while 
    identifying the RDF triples: 

    1. **Verb Form**: Use consistent verb forms (preferably base form) for predicates, e.g., "support", "love", 
    "dislike". 2. **Specific Relationships**: Identify specific, directional verbs that clearly indicate the nature 
    of the relationship between the subject and the object, e.g., "is a parent of", "works at". 3. **Invertable Relationships**: 
    if the relationship is able to be inverted, please generate the inverse. 4. Make sure all triples and thier inverses are logically consistent

    Please proceed with generating the knowledge graph based on the conversation provided.
    """

    request_prompt = """\nPlease get a list of RDF triples (subject, predicate, object) and the appropriate inverted triples, from this conversation.
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

    message = chat_llm(
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

def triples_to_embeddings(triples):
    # Split the triples into separate lists for subjects, verbs, and objects
    subjects = [triple[0] for triple in triples]
    verbs = [triple[1] for triple in triples]
    objects = [triple[2] for triple in triples]

    # Encode the subjects, verbs, and objects
    subject_embeddings = embedding_model.encode(subjects)
    verb_embeddings = embedding_model.encode(verbs)
    object_embeddings = embedding_model.encode(objects)

    # arrange the embeddings into a list of tuples
    svo_embeddings = list(zip(subject_embeddings, verb_embeddings, object_embeddings))

    return svo_embeddings


def build_graph(subject, svo_embeddings, svo_text, similarity_threshold, depth):
    collected_triples = []
    visited = set()

    def dfs(current_subject, current_depth):
        if current_depth == depth:
            return
        visited.add(current_subject)
        # Finding all verb-object pairs for the current subject
        for svo_emb, svo_txt in zip(svo_embeddings, svo_text):
            if svo_txt[0] == current_subject:
                verb, object_ = svo_txt[1], svo_txt[2]
                collected_triples.append((current_subject, verb, object_))
                # Finding the most similar subject to the current object
                max_similarity = 0
                most_similar_subject = None
                for candidate_emb, candidate_txt in zip(svo_embeddings, svo_text):
                    if candidate_txt[0] != current_subject and candidate_txt[0] not in visited:
                        similarity = cosine_similarity(
                            svo_emb[2].reshape(1, -1), candidate_emb[0].reshape(1, -1)
                        )[0][0]
                        if similarity > max_similarity and similarity >= similarity_threshold:
                            max_similarity = similarity
                            most_similar_subject = candidate_txt[0]
                if most_similar_subject:
                    dfs(most_similar_subject, current_depth + 1)

    dfs(subject, 0)
    return collected_triples

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

svo_text = extract_triples(text)
svo_embeddings = triples_to_embeddings(svo_text)
print(svo_text)

graph = build_graph("Ragna", svo_embeddings, svo_text, 0.80, 4)
print(graph)

# Extract embeddings of the first and second SVO triples
subject_embedding_1, verb_embedding_1, object_embedding_1 = svo_embeddings[0]
subject_embedding_2, verb_embedding_2, object_embedding_2 = svo_embeddings[1]

# Compute cosine similarity between corresponding parts
subject_similarity = cosine_similarity(subject_embedding_1.reshape(1, -1), subject_embedding_2.reshape(1, -1))[0][0]
verb_similarity = cosine_similarity(verb_embedding_1.reshape(1, -1), verb_embedding_2.reshape(1, -1))[0][0]
object_similarity = cosine_similarity(object_embedding_1.reshape(1, -1), object_embedding_2.reshape(1, -1))[0][0]

# Print out the similarity
print(f"Cosine similarity between subjects 1 and 2: {subject_similarity:.2f}")
print(f"Cosine similarity between verbs 1 and 2: {verb_similarity:.2f}")
print(f"Cosine similarity between objects 1 and 2: {object_similarity:.2f}")


subject_embedding_3, verb_embedding_3, object_embedding_3 = svo_embeddings[2]

# Compute cosine similarity between corresponding parts
object_1_to_subject_3_similarity = cosine_similarity(object_embedding_1.reshape(1, -1), subject_embedding_3.reshape(1, -1))[0][0]
verb_similarity = cosine_similarity(verb_embedding_2.reshape(1, -1), verb_embedding_3.reshape(1, -1))[0][0]
object_similarity = cosine_similarity(object_embedding_2.reshape(1, -1), object_embedding_3.reshape(1, -1))[0][0]

# Print out the similarity
print(f"Cosine similarity between object_1 and subject_3: {object_1_to_subject_3_similarity:.2f}")
print(f"Cosine similarity between verbs 2 and 3: {verb_similarity:.2f}")
print(f"Cosine similarity between objects 2 and 3: {object_similarity:.2f}")