import json
from collections import defaultdict


class Vertex:
    def __init__(self, id, properties=None):
        self.id = id
        self.properties = properties or {}

    def add_property(self, key, value):
        self.properties[key] = value

    def get_property(self, key):
        return self.properties.get(key)


class GraphStore:
    def __init__(self, name=None):
        self.vertices = {}
        self.edges = defaultdict(list)
        if name:
            self.name = name
            # if the name is provided attempt to load the graph from file
            self.load_from_file(f"{self.name}.json")
        else:
            self.name = "Default"

    def add_vertex(self, id, properties=None):
        self.vertices[id] = Vertex(id, properties)

    def get_vertex(self, id):
        return self.vertices.get(id)

    def get_all_entities(self):
        return list(self.vertices.keys())

    def get_all_predicates(self):
        predicates = set()
        for edges in self.edges.values():
            for edge in edges:
                predicates.add(edge['predicate'])
        return list(predicates)

    def add_edge(self, id1, predicate, id2):
        if id1 not in self.vertices:
            self.add_vertex(id1)
        if id2 not in self.vertices:
            self.add_vertex(id2)

        vertex1 = self.get_vertex(id1)
        vertex2 = self.get_vertex(id2)

        # Check to avoid duplicate edges
        if not any(edge['vertex'].id == id2 and edge['predicate'] == predicate for edge in self.edges[id1]):
            self.edges[id1].append({'vertex': vertex2, 'predicate': predicate})
            self.edges[id2].append({'vertex': vertex1, 'predicate': predicate})  # Assuming undirected graph


    def remove_edge(self, id1, predicate, id2, remove_orphans=False):
        if id1 in self.edges:
            self.edges[id1] = [edge for edge in self.edges[id1] if
                               not (edge['vertex'].id == id2 and edge['predicate'] == predicate)]
        if id2 in self.edges:
            self.edges[id2] = [edge for edge in self.edges[id2] if
                               not (edge['vertex'].id == id1 and edge['predicate'] == predicate)]

        if remove_orphans:
            if not self.edges.get(id1):
                self.remove_entity(id1)
            if not self.edges.get(id2):
                self.remove_entity(id2)

    def remove_entity(self, id):
        if id in self.vertices:
            del self.vertices[id]
        if id in self.edges:
            del self.edges[id]

        for edges in self.edges.values():
            edges[:] = [edge for edge in edges if edge['vertex'].id != id]

    def merge_entities(self, id_to_keep, id_to_merge):
        if id_to_merge in self.vertices:
            if id_to_keep not in self.vertices:
                self.add_vertex(id_to_keep)

            for edges in self.edges.values():
                for edge in edges:
                    if edge['vertex'].id == id_to_merge:
                        edge['vertex'] = self.vertices[id_to_keep]

            if id_to_merge in self.edges:
                if id_to_keep not in self.edges:
                    self.edges[id_to_keep] = []

                self.edges[id_to_keep].extend(self.edges[id_to_merge])
                del self.edges[id_to_merge]

            del self.vertices[id_to_merge]

    def merge_predicates(self, predicate_to_keep, predicate_to_merge):
        for edge_list in self.edges.values():
            for edge in edge_list:
                if edge['predicate'] == predicate_to_merge:
                    edge['predicate'] = predicate_to_keep

    def remove_predicate(self, predicate):
        for edges in self.edges.values():
            edges[:] = [edge for edge in edges if edge['predicate'] != predicate]

    def get_network_string(self, id):
        network_str = ""
        if id in self.edges:
            predicates = defaultdict(list)
            for edge in self.edges[id]:
                predicates[edge['predicate']].append(edge['vertex'].id)

            if predicates:  # Check if there are any predicates
                network_str = f"{id}\n"
                for predicate, target_vertices in predicates.items():
                    # Filtering out None values before joining
                    target_vertices = [str(v) for v in target_vertices if v is not None]
                    network_str += f"  {predicate}: " + ", ".join(target_vertices) + "\n"

        return network_str

    def save_to_file(self, filename=None):
        if not filename:
            filename = f"{self.name}.json"
        graph_data = {
            'vertices': {id: vertex.properties for id, vertex in self.vertices.items()},
            'edges': {id: [{'vertex': edge['vertex'].id, 'predicate': edge['predicate']} for edge in edges] for
                      id, edges in self.edges.items()}
        }
        with open(filename, 'w') as file:
            json.dump(graph_data, file)

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                graph_data = json.load(file)

            self.vertices = {id: Vertex(id, properties) for id, properties in graph_data['vertices'].items()}
            for id, edges in graph_data['edges'].items():
                for edge in edges:
                    self.add_edge(id, edge['predicate'], edge['vertex'])
        except FileNotFoundError:
            print(f"File {filename} not found, creating new graph")

    def process_instructions(self, instructions_json):
        try:
            instructions = json.loads(instructions_json)

            # Merge specified entities
            for merge_info in instructions.get('entities_to_merge', []):
                self.merge_entities(merge_info['id_to_keep'], merge_info['id_to_merge'])

            # Merge specified predicates
            for merge_info in instructions.get('predicates_to_merge', []):
                self.merge_predicates(merge_info['id_to_keep'], merge_info['id_to_merge'])

            # Remove specified entities
            for entity in instructions.get('entities_to_remove', []):
                self.remove_entity(entity)

            # Remove specified predicates
            for predicate in instructions.get('predicates_to_remove', []):
                self.remove_predicate(predicate)

            # Remove specified links
            for link_info in instructions.get('links_to_remove', []):
                self.remove_edge(link_info['id_1'], link_info['predicate'], link_info['id_2'])

        except json.JSONDecodeError:
            print("Invalid JSON format")


# Example usage:
if __name__ == '__main__':
    graph = GraphStore()

    graph.add_edge('Joey', 'Loves', 'Heather')
    graph.add_edge('Joey', 'Loves', 'Cheese')
    graph.add_edge('Joey', 'Dislikes', 'Mosquitoes')

    # Save to file
    graph.save_to_file('graph_data.json')

    # Load from file
    new_graph = GraphStore()
    new_graph.load_from_file('graph_data.json')
    print(new_graph.get_network_string('Joey'))
