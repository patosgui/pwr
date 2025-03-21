#!/usr/bin/env python3

# Copyright Muxup contributors.
# Distributed under the terms of the MIT license, see LICENSE for details.
# SPDX-License-Identifier: MIT

import streamlit as st
from streamlit_authenticator.authenticate import Authenticate
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Callable, TypedDict

import requests
from bs4 import BeautifulSoup
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("pwr.log")],
)
logger = logging.getLogger(__name__)

### Type definitions ###

URLAndTitleList = list[list[str]]


class SourceData(TypedDict):
    seen: list[str]
    entries: URLAndTitleList


class PWRData(TypedDict):
    last_read: str
    last_fetch: str
    last_filter: str
    sources: dict[str, SourceData]


### Config ###


def get_sources() -> (
    dict[str, tuple[Callable[..., URLAndTitleList], *tuple[object, ...]]]
):
    return {
        "Hacker News": (feed_fetcher, "https://news.ycombinator.com/rss", True),
        "CNBC": (
            feed_fetcher,
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
        ),
        "Publico": (feed_fetcher, "https://feeds.feedburner.com/PublicoRSS"),
        "lobste.rs": (feed_fetcher, "https://lobste.rs/rss", True),
        "/r/programminglanguages": (
            feed_fetcher,
            "http://www.reddit.com/r/programminglanguages/.rss",
        ),
        "/r/LLVM": (feed_fetcher, "http://www.reddit.com/r/LLVM/top.rss?t=week"),
        "MLIR": (discourse_fetcher, "https://discourse.llvm.org/c/mlir/31"),
        "/r/cpp": (feed_fetcher, "http://www.reddit.com/r/cpp/top.rss?t=week"),
        "/r/LocalLLaMA": (
            feed_fetcher,
            "http://www.reddit.com/r/LocalLLaMA/top.rss?t=week",
        ),
        "Ars Technica": (
            feed_fetcher,
            "https://feeds.arstechnica.com/arstechnica/index",
        ),
        "cs.PL": (arxiv_fetcher, "cs.PL"),
        "cs.AR": (arxiv_fetcher, "cs.AR"),
    }


# URLs that will be opened unconditionally when performing the 'read' action.
data_dir = Path.home() / ".local" / "share" / "pwr"
saved_seen_url_limit = 250
fetch_timeout = 10
read_url_batch_size = 10

### Fetchers ###

# A fetcher is passed whatever arguments were present in the sources
# dictionary, and is responsible for returning an array of [url, title]
# arrays. The caller will take care of removing any [url, title] entries where
# the url has already been seen. A fetcher might append `##someval` to a URL
# to force it appear fresh (where someval might be a timestamp, number of
# replies, or some other data).


def feed_fetcher(url: str, comments_as_link: bool = False) -> URLAndTitleList:
    soup = BeautifulSoup(fetch_from_url(url), features="xml")
    entries = soup.find_all(["item", "entry"])
    extracted = []
    for entry in entries:
        title = entry.find("title").text
        link = entry.find(comments_as_link and "comments" or "link")
        url = link.get("href") or link.text
        extracted.append([url, title])
    return extracted


# Custom fetcher rather than just using the RSS feed because the arXiv RSS
# feeds only include any papers posted in the last day, so it's possible to
# miss them if you don't fetch regularly enough.
def arxiv_fetcher(category: str) -> URLAndTitleList:
    url = f"https://arxiv.org/list/{category}/recent"
    soup = BeautifulSoup(fetch_from_url(url), "html.parser")
    extracted = []

    for dt in soup.find_all("dt"):
        title_tag = dt.find_next("div", class_="list-title")
        title = title_tag.text.replace("Title:", "").strip()
        abstract = dt.find_next("a", title="Abstract")
        url = "https://arxiv.org" + abstract["href"]
        extracted.append([url, title])
    return extracted


# The Discourse RSS feeds don't provide the same listing as when viewing a
# category sorted by most recently replied to (see
# <https://meta.discourse.org/t/missing-rss-feed-which-corresponds-to-new-topics/295686>), so extract it ourselves.
def discourse_fetcher(url: str) -> URLAndTitleList:
    soup = BeautifulSoup(fetch_from_url(url), "html.parser")
    extracted = []

    topics = soup.find_all("tr", class_="topic-list-item")
    for topic in topics:
        title_tag = topic.find("a", class_="title")
        title = title_tag.text.strip()
        url = title_tag.get("href")
        replies = topic.find("span", class_="posts").text.strip()
        extracted.append([f"{url}##{replies}", append_replies(title, int(replies))])
    return extracted


def ghdiscussions_fetcher(ghpath: str) -> URLAndTitleList:
    soup = BeautifulSoup(
        fetch_from_url(f"https://github.com/{ghpath}/discussions"), "html.parser"
    )
    extracted = []

    for topic in soup.find_all("a", attrs={"data-hovercard-type": "discussion"}):
        title = topic.text.strip()
        url = f"https://github.com{topic.get('href')}"
        replies_element = topic.find_next(
            "a", attrs={"aria-label": lambda x: x and "comment" in x}
        )
        replies = replies_element.text.strip()
        extracted.append([f"{url}##{replies}", append_replies(title, int(replies))])
    return extracted


def groupsio_fetcher(groupurl: str) -> URLAndTitleList:
    soup = BeautifulSoup(fetch_from_url(groupurl), "html.parser")
    extracted = []
    for topic_span in soup.find_all("span", class_="subject"):
        link = topic_span.find("a")
        title = link.text.strip()
        url = link.get("href")
        reply_count_span = topic_span.find("span", class_="hashtag-position")
        reply_count = reply_count_span.text.strip() if reply_count_span else "0"
        extracted.append(
            [f"{url}##{reply_count}", append_replies(title, int(reply_count))]
        )
    return extracted


### Helper functions ###


def append_replies(title: str, count: int) -> str:
    if count == 1:
        return f"{title} (1 reply)"
    return f"{title} ({count} replies)"


def fetch_from_url(url: str, max_retries: int = 5, delay: int = 1) -> str:
    headers = {"User-Agent": "pwr - paced web reader"}
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {url}...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info("Done")
            return response.text
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 4
    logger.error("Max retries reached. Giving up.")
    sys.exit(1)


def extract_article_content(url: str) -> str:
    """Extract the main content from a news article URL"""
    try:
        html = fetch_from_url(url)
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()

        # Get text from paragraphs - focusing on article content
        paragraphs = []
        if soup.find("article"):
            article = soup.find("article")
            paragraphs = article.find_all("p")
        else:
            # Fallback to main content area or just all paragraphs
            main_content = soup.find(
                class_=lambda x: x
                and ("content" in x.lower() or "article" in x.lower())
            )
            if main_content:
                paragraphs = main_content.find_all("p")
            else:
                paragraphs = soup.find_all("p")

        content = "\n\n".join(
            [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50]
        )
        return content
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return f"Error extracting content: {str(e)}"


def summarize_with_claude(content: str, title: str, claude_api_key: str | None) -> str:
    """Summarize the article content using Claude API"""
    try:
        # Get API key from environment or config
        api_key = claude_api_key
        if not api_key:
            return (
                "Claude API key not found. Set the CLAUDE_API_KEY environment variable."
            )

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""
        Summarize the following article titled "{title}" in 3-4 concise bullet points:
        
        {content[:15000]}  # Limit content to avoid token limits
        
        Focus on the main points and key takeaways. Be concise and informative.
        """

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.0,
            system="You are a helpful assistant that summarizes news articles accurately and concisely.",
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text
    except Exception as e:
        return f"Error generating summary: {str(e)}"


class URLCache:
    """
    A mini cache for storing URLs that have already been fetched.
    Example:
    {
        "urls": {
            "HN": [
                "https://news.ycombinator.com/item?id=xxxxxxxx",
                "https://news.ycombinator.com/item?id=xxxxxxxx"
            ]
        }
    }
    """

    filename: Path
    data: dict[str, list[str]]
    max_entries: int

    def __init__(self, filename, max_entries=500):
        self.filename = filename
        self.data = dict()
        self.max_entries = max_entries

        if self.filename.exists():
            logger.info(f"Loading URLs from cache at {self.filename}")
            with open(self.filename, "r") as f:
                self.data = json.load(f)
        else:
            logger.info(f"Created new cache at {self.filename}")
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filename, "w") as f:
                self.data["urls"] = {}
                self.filename.write_text(json.dumps(self.data))

    def save(self) -> None:
        self.filename.write_text(json.dumps(self.data, indent=4))

    def add(self, source, url: str) -> bool:
        if source in self.data["urls"]:
            url_list = self.data["urls"][source]
            if url in url_list:
                return False
            if len(url_list) >= self.max_entries:
                url_list.pop(0)
            url_list.append(url)
        else:
            self.data["urls"][source] = [url]

        self.save()  # This is heavy but performance is not an issue
        return True


def do_fetch(url_cache: URLCache):
    # [[url,title,source_name]]
    returned_urls: list[list[str]] = []
    total_filtered_extracted = 0
    for source_name, source_info in get_sources().items():
        logger.info(f"Processing source {source_name}")

        func = source_info[0]
        extracted = func(*source_info[1:])

        filtered = [
            [source_name, title, url]
            for url, title in extracted
            if url_cache.add(source_name, url)
        ]

        returned_urls.extend(filtered)
        total_filtered_extracted += len(filtered)
        logger.info(
            f"Retrieved {len(extracted)} items, {len(filtered)} remain after removing seen items"
        )

    logger.info(
        f"\nA total of {total_filtered_extracted} items were queued up for filtering."
    )

    return pd.DataFrame(returned_urls, columns=["Source", "Title", "URL"])


def get_selected_items(key: str):
    return
    st.write(st.session_state[key])


def summarize_article_callback(url: str, title: str, claude_api_key: str | None):
    """Callback function for the summarize button"""
    with st.spinner(f"Summarizing article: {title[:30]}..."):
        # Get article content
        content = extract_article_content(url)

        # Summarize with Claude
        summary = summarize_with_claude(content, title, claude_api_key)

        # Store summary in session state to persist between reruns
        if "summaries" not in st.session_state:
            st.session_state.summaries = {}

        st.session_state.summaries[url] = summary


def summarize_source_callback(
    source: str, articles: list[tuple[str, str]], claude_api_key: str | None
):
    """Callback function for summarizing all articles from a source"""
    with st.spinner(f"Summarizing all articles from {source}..."):
        # Initialize source summaries in session state if not exists
        if "source_summaries" not in st.session_state:
            st.session_state.source_summaries = {}

        # Collect all article summaries
        all_summaries = []
        for url, title in articles:
            # If we don't have a summary for this article yet, get one
            if url not in st.session_state.summaries:
                content = extract_article_content(url)
                summary = summarize_with_claude(content, title, claude_api_key)
                st.session_state.summaries[url] = summary
            all_summaries.append(st.session_state.summaries[url])

        # Create a prompt for Claude to summarize all the summaries
        prompt = f"""
        Create a concise summary of the following news articles from {source}. 
        Focus on the main themes, trends, and key takeaways across all articles:

        {'\n\n'.join(all_summaries)}
        """

        try:
            client = anthropic.Anthropic(api_key=claude_api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.0,
                system="You are a helpful assistant that creates concise summaries of multiple news articles, focusing on common themes and key takeaways.",
                messages=[{"role": "user", "content": prompt}],
            )
            st.session_state.source_summaries[source] = message.content[0].text
        except Exception as e:
            st.session_state.source_summaries[
                source
            ] = f"Error generating source summary: {str(e)}"


def streamlit_data_editors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the dataframes that are generated by streamlit to keep the data alive
    """
    sources = df["Source"].unique()

    # Initialize summaries in session state if not exists
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "source_summaries" not in st.session_state:
        st.session_state.source_summaries = {}

    # Apply the make_clickable function to the link column
    logger.debug(f"DataFrame contents:\n{df}")

    for source in sources:
        # Make a subheader for each source
        st.subheader(source)

        source_df = df.query("Source == @source").drop(columns=["Source"])

        # Add "Summarize All" button for this source
        if st.button(f"Summarize All {source} Articles", key=f"summarize_all_{source}"):
            # Get list of (url, title) tuples for this source
            articles = list(zip(source_df["URL"], source_df["Title"]))
            summarize_source_callback(source, articles, claude_api_key)

        # Display source summary if it exists
        if source in st.session_state.source_summaries:
            with st.expander(f"View {source} Summary", expanded=True):
                st.markdown(st.session_state.source_summaries[source])

        # Add buttons separately for each row
        for i, row in source_df.iterrows():
            url = row["URL"]
            title = row["Title"]

            # Create a unique key for each button
            button_key = f"summarize_btn_{source}_{i}"

            # Create a container for each article for better organization
            with st.container():
                # Create columns for the title and button
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Display title with link
                    st.markdown(f"[{title}]({url})")

                with col2:
                    # Create a button for this article
                    if st.button("Summarize with Claude", key=button_key):
                        assert claude_api_key
                        summarize_article_callback(url, title, claude_api_key)

                # If we have a summary for this URL, display it
                if url in st.session_state.summaries:
                    with st.expander("View Summary", expanded=True):
                        st.markdown(st.session_state.summaries[url])

                # Add a separator between articles
                st.markdown("---")


def load(file: Path):
    if not file.exists():
        return None

    with open(file, "r") as f:
        data = json.load(f)
    result = pd.DataFrame()

    for source, entry in data["urls"].items():
        for e in entry:
            e["Source"] = source
        df = pd.DataFrame(entry)
        result = pd.concat([result, df], ignore_index=True)

    return result


def save(data, file: Path):
    json_data: dict[str, dict[str, list[str]]] = dict()
    json_data["urls"] = {}

    if data.empty:
        file.write_text(json.dumps(json_data, indent=4))
        return

    sources = data["Source"].unique()
    for source in sources:
        source_df = data[data.Source == source].copy()
        source_df = source_df.drop(columns=["Source"])
        json_data["urls"][source] = source_df.to_dict("records")

    file.write_text(json.dumps(json_data, indent=4))


### Main ###

if __name__ == "__main__":
    logger.info("Running session")

    state_file = Path.home() / ".local" / "share" / "pwr" / "state.json"
    cache_file = Path.home() / ".local" / "share" / "pwr" / "pwrcache.json"
    logger.debug(f"Cache file path: {cache_file}")

    # Create the data dir for the cache and the log in confi
    data_dir.mkdir(parents=True, exist_ok=True)

    config_dir = (
        Path(os.environ["CONFIG_DIR"]) if os.getenv("CONFIG_DIR", None) else data_dir
    )
    logger.info(f"Searching for login data in {config_dir}")
    with open(config_dir / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    claude_api_key = None
    with open(config_dir / "claude-key.txt", "r") as f:
        claude_api_key = f.read().strip()

    if not claude_api_key:
        logger.warning(
            "Claude API key not found. Set the CLAUDE_API_KEY environment variable to enable article summarization."
        )

    # Start the login page
    authenticator = Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["pre-authorized"],
    )
    name, authentication_status, username = authenticator.login()

    # Deal with bad login
    if authentication_status is False:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")

    # Block access to the data if login failed
    if not authentication_status:
        sys.exit(0)

    if "data" not in st.session_state:
        st.session_state["data"] = None

    # Datafram with Title, URL, Source, Summary
    data = None

    _, right_col = st.columns([3, 1])

    with right_col:
        # Only update data upon explicit user request
        if st.button("Refresh Data"):
            data = do_fetch(URLCache(cache_file))

    if data is None:
        if st.session_state.data is None:
            data = load(state_file)
            if data is None:
                data = do_fetch(URLCache(cache_file))
        else:
            data = st.session_state.data

    assert data is not None, "Data is none!"

    save(data, state_file)

    # No new data, exit early
    if data.empty:
        with st.chat_message("user"):
            st.write("Hey👋! You're up to date!")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    st.session_state.data = data

    streamlit_data_editors(data)

    sys.stdout.flush()
    sys.stderr.flush()
