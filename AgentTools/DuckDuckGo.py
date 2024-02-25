from duckduckgo_search import DDGS

def duckduckgo_search(query, max_results=5):
    """Searches the web and returns a list of results."""
    with DDGS() as ddgs:
        results = [result for result in ddgs.text(query, max_results=max_results)]
    return results