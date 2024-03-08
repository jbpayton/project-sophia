import html2text
import requests

class WebpageTextBrowser:

    def __init__(self, url, lines_per_page=30):
        self.LINES_PER_PAGE = lines_per_page
        self.url = url
        self.text_lines = []
        self.current_line = 0
        self.total_lines = 0
        self.search_results = []
        self.current_search_result_index = -1
        self._fetch_and_prepare_text()

    def _fetch_and_prepare_text(self):
        """Fetches the webpage content and prepares the text."""
        response = requests.get(self.url)
        html_content = response.text

        h = html2text.HTML2Text()
        h.ignore_links = False
        text = h.handle(html_content)

        self.text_lines = text.split('\n')
        self.total_lines = len(self.text_lines)

    def _construct_page_content(self, page, is_search_result=False):
        """Constructs the page content with header and footer."""
        header = f"URL: {self.url} - Top of the Browser\n---\n"
        footer = "\n---\n"
        if is_search_result:
            footer += f"Search result {self.current_search_result_index + 1} of {len(self.search_results)}"
        else:
            # Adjust the calculation of the current_page_number here

            current_page_number = ((self.current_line + 1) // self.LINES_PER_PAGE)
            if self.current_line == 0:
                current_page_number = 1


            total_pages = self.total_lines // self.LINES_PER_PAGE
            if total_pages == 0:
                total_pages = 1
            footer += f"Page {current_page_number} of {total_pages}"

        # Combine header, page, and footer
        page_text = '\n'.join(page)
        return f"{header}{page_text}{footer}"

    def get_text_page(self, start_line=None, is_search_result=False):
        """Returns a page of text starting from a specific line or the current line."""
        page_length = self.LINES_PER_PAGE
        if start_line is not None:
            self.current_line = start_line
        end_line = self.current_line + page_length
        page = self.text_lines[self.current_line:end_line]
        # Avoid advancing the current line if displaying a search result
        if not is_search_result:
            # Ensure we don't go past the end
            if end_line < self.total_lines:
                self.current_line = end_line

        return self._construct_page_content(page, is_search_result)

    def search_text(self, search_term):
        """Searches the text for a term and stores the lines and their numbers containing that term, case insensitively."""
        lower_search_term = search_term.lower()  # Convert the search term to lower case
        self.search_results = [(i, line) for i, line in enumerate(self.text_lines) if lower_search_term in line.lower()]
        return len(self.search_results)  # Return the number of results found

    def go_to_search_result(self, result_index):
        """Navigates to and displays the search result."""
        if 0 <= result_index < len(self.search_results):
            line_number, _ = self.search_results[result_index]
            self.current_search_result_index = result_index
            # Center the search result in the page if possible
            half_page = self.LINES_PER_PAGE // 2
            start_line = max(line_number - half_page, 0)
            # Directly return the page with the search result
            return self.get_text_page(start_line=start_line, is_search_result=True)
        else:
            return "Invalid search result index."

    def next_page(self):
        """Advances to the next page of text."""

        # make sure that if we are just coming from a search result, we snap to a page boundary
        if self.current_search_result_index != -1:
            # we need to make sure the current line is a multiple of LINES_PER_PAGE
            self.current_line = (self.current_line // self.LINES_PER_PAGE) * self.LINES_PER_PAGE
            self.current_search_result_index = -1

        return self.get_text_page()

    def previous_page(self):
        """Goes back to the previous page of text."""
        self.current_line = max(self.current_line - 2 * self.LINES_PER_PAGE, 0)
        return self.get_text_page()

    def next_search_result(self):
        """Navigates to the next search result."""
        if self.search_results:
            self.current_search_result_index = (self.current_search_result_index + 1) % len(self.search_results)
            return self.go_to_search_result(self.current_search_result_index)
        else:
            return "No search results."

    def previous_search_result(self):
        """Navigates to the previous search result."""
        if self.search_results:
            self.current_search_result_index = (self.current_search_result_index - 1 + len(self.search_results)) % len(self.search_results)
            return self.go_to_search_result(self.current_search_result_index)
        else:
            return "No search results."


if __name__ == "__main__":
    browser = WebpageTextBrowser("https://blazblue.fandom.com/wiki/Rachel_Alucard")
    #browser = WebpageTextBrowser("https://www2.latech.edu/~acm/helloworld/HTML.html")
    print(browser.get_text_page())
    print(browser.next_page())
    print(browser.previous_page())
    print(browser.search_text("World"))
    print(browser.go_to_search_result(0))
    print(browser.next_search_result())
    while True:
        # on enter, go to the next page of text
        input()
        print(browser.next_page())
