import wikipedia

def search_wikipedia(query):
    """Searches Wikipedia for a given query and returns a summary."""
    output = wikipedia.summary(query, auto_suggest=False)
    return output
