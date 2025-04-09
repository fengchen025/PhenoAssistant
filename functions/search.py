r"""
This module provides functions to scrape web pages and extract their content, including titles, headers, and paragraphs.
It supports only static content scraping for the moment, but a work-in-progress function is available for scraping dynamic content using pyppeteer.
Functions:
        scrape_url(url: str) -> dict:
        fetch_url(url: str) -> requests.Response:
                Fetches the content of the URL.
                        url (str): The URL to fetch.
                        requests.Response: The response object containing the content of the URL.
        extract_title(soup: BeautifulSoup) -> str:
                Extracts the title from the BeautifulSoup object.
                        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML.
                        str: The title of the web page, or "No Title Found" if no title is present.
        extract_content_structure(soup: BeautifulSoup) -> list:
                Extracts the content structure from the BeautifulSoup object.
                        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML.
                        list: A list of dictionaries, each containing a header and its associated paragraphs.
        scrape_urls(urls: Union[str, List[str]]) -> dict:
        scrape_single_url_dynamic(url): #! WIP
                Scrapes a single URL with dynamic content using pyppeteer.
                                - "content" (list): A list of dictionaries, each containing a header and its associated paragraphs.
"""
import asyncio
from typing import List, Union

import requests
from bs4 import BeautifulSoup
from pyppeteer import launch

from googlesearch import search

def scrape_url(url: str) -> dict:
    """
    Scrapes the given URL and extracts the page title, headers (h1, h2, h3), and paragraphs.
    Args:
                    url (str): The URL of the web page to scrape.
    Returns:
                    dict: A dictionary containing the following keys:
                                    - "url" (str): The URL of the web page.
                                    - "title" (str): The title of the web page, or "No Title Found" if no title is present.
                                    - "headers" (dict): A dictionary with keys "h1", "h2", and "h3", each containing a list of text from the respective headers.
                                    - "paragraphs" (list): A list of text from all paragraph elements.
                                    - "error" (str, optional): An error message if the request to the URL failed.
    """
    response = fetch_url(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        title = extract_title(soup)
        content_structure = extract_content_structure(soup)
        return {"url": url, "title": title, "content": content_structure}
    else:
        return {
            "url": url,
            "error": f"Failed to retrieve content. Status code: {response.status_code}",
        }


def fetch_url(url: str) -> requests.Response:
    """Fetches the content of the URL."""
    return requests.get(url)


def extract_title(soup: BeautifulSoup) -> str:
    """Extracts the title from the BeautifulSoup object."""
    return soup.title.string if soup.title else "No Title Found"


def extract_content_structure(soup: BeautifulSoup) -> list:
    """Extracts the content structure from the BeautifulSoup object."""
    content_structure = []
    current_header = None
    for element in soup.find_all(["h1", "h2", "h3", "p"]):
        if element.name in ["h1", "h2", "h3"]:
            current_header = {"header": element.text.strip(), "paragraphs": []}
            content_structure.append(current_header)
        elif element.name == "p" and current_header:
            current_header["paragraphs"].append(element.text.strip())
    return content_structure


def scrape_urls(urls: Union[str, List[str]]) -> dict:
    """
    Scrapes data from a list of URLs.

    Args:
                    urls (str or list of str): A single URL string or a list of URL strings to scrape.

    Returns:
                    dict: A dictionary where the keys are the URLs and the values are the scraped data.
    """
    # Ensure urls is a list (if a single URL is passed, convert it into a list)
    if isinstance(urls, str):
        urls = [urls]  # Convert a single URL string into a list

    # Scrape each URL and gather results
    results = dict()
    for url in urls:
        scraped_data = scrape_url(url)
        if scraped_data:
            results[url] = scraped_data

    return results

def scrape_single_url_dynamic(url):
    """
    Scrapes the content of a single URL dynamically using a headless browser.

    This function uses an asynchronous approach to launch a headless browser,
    navigate to the specified URL, and retrieve the page content. The content
    is then parsed using BeautifulSoup to extract the title and a structured
    representation of headers and paragraphs.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        dict: A dictionary containing the URL, the title of the page, and a
              structured representation of the content with headers and
              associated paragraphs. The structure is as follows:
              {
                  "url": str,
                  "title": str,
                  "content": [
                      {
                          "header": str,
                          "paragraphs": [str, ...]
                      },
                      ...
                  ]
              }
    """
    async def main(url):
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.goto(url, {"waitUntil": "networkidle2"})
        content = await page.content()
        await browser.close()
        return content

    content = asyncio.get_event_loop().run_until_complete(main(url))
    soup = BeautifulSoup(content, "html.parser")

    # Proceed with the same parsing as before
    title = soup.title.string if soup.title else "No Title Found"
    content_structure = []
    current_header = None

    for element in soup.find_all(["h1", "h2", "h3", "p"]):
        if element.name in ["h1", "h2", "h3"]:
            current_header = {"header": element.text.strip(), "paragraphs": []}
            content_structure.append(current_header)
        elif element.name == "p" and current_header:
            current_header["paragraphs"].append(element.text.strip())

    return {"url": url, "title": title, "content": content_structure}

#!====================================================================================================
 
def google_search(query: str, num_results: int = 10) -> List[str]:
    """
    Perform a Google search and return a list of URLs.
    Args:
                    query (str): The search query string.
                    num_results (int, optional): The number of search results to return. Defaults to 3.
    Returns:
                    List[str]: A list of URLs from the search results.
    """

    links = []
    # for j in search(query, num=num_results, stop=num_results):
    for j in search(query, num_results=num_results):
        links.append(j)

    return links

def search_and_scrape(query: str, num_results: int = 10) -> dict:
    """
    Perform a Google search with the given query and scrape the top search results.
    Args:
                    query (str): The search query string. e.g. "plant phenotyping"
                    num_results (int, optional): The number of search results to scrape. Defaults to 10.
    Returns:
                    dict: A dictionary containing the searched urls and the scraped contents.
    """
    links = google_search(query, num_results)
    scraped_data = scrape_urls(links)
    return scraped_data

# Example usage:
if __name__ == "__main__":
    query = "Plant phenotyping"
    num_results = 10
    content = search_and_scrape(query, num_results)
    print(content)