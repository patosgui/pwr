#!/usr/bin/env python3

# Copyright Muxup contributors.
# Distributed under the terms of the MIT license, see LICENSE for details.
# SPDX-License-Identifier: MIT

import streamlit as st
import datetime
import pandas as pd
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, TypedDict

import requests
from bs4 import BeautifulSoup

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

# Wrapped in a function to workaround Python's lack of support for forward
# declarations of functions (such as the referenced fetcher functions)
# fmt: off
def get_sources() -> dict[str, tuple[Callable[..., URLAndTitleList], *tuple[object, ...]]]:
    return {
        #"Rust Internals": (discourse_fetcher, "https://internals.rust-lang.org/"),
        "HN": (feed_fetcher, "https://news.ycombinator.com/rss", True),
        #"lobste.rs": (feed_fetcher, "https://lobste.rs/rss", True),
        "/r/programminglanguages": (feed_fetcher, "http://www.reddit.com/r/programminglanguages/.rss"),
        "/r/rust": (feed_fetcher, "http://www.reddit.com/r/rust/top.rss?t=week"),
        "Muxup": (feed_fetcher, "https://muxup.com/feed.xml"),
        "Igalia": (feed_fetcher, "https://www.igalia.com/feed.xml"),
        "cs.PL": (arxiv_fetcher, "cs.PL"),
        "cs.AR": (arxiv_fetcher, "cs.AR"),
        "Hylo discussions": (ghdiscussions_fetcher, "orgs/hylo-lang"),
        "RISC-V announcements": (groupsio_fetcher, "https://lists.riscv.org/g/tech-announce/topics"),
        "Nim forum": (nimforum_fetcher,),
    }
# fmt: on


# URLs that will be opened unconditionally when performing the 'read' action.
extra_urls_for_read = ["https://guardian.co.uk"]
data_dir = Path(
    os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share" / "pwr")
)
data_file = data_dir / "data.json"
preferred_browser = os.environ.get("BROWSER", "firefox")
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


def nimforum_fetcher() -> URLAndTitleList:
    thread_data = json.loads(fetch_from_url("https://forum.nim-lang.org/threads.json"))
    extracted = []
    for thread in thread_data["threads"]:
        extracted.append(
            [
                f"https://forum.nim-lang.org/t/{thread['id']}##{thread['replies']}",
                append_replies(thread["topic"], thread["replies"]),
            ]
        )
    return extracted


### Helper functions ###


def append_replies(title: str, count: int) -> str:
    if count == 1:
        return f"{title} (1 reply)"
    return f"{title} ({count} replies)"


def load_data() -> PWRData:
    # if data_file.exists():
    #    return json.loads(data_file.read_text())  # type: ignore
    return {
        "last_read": "Never",
        "last_fetch": "Never",
        "last_filter": "Never",
        "sources": {},
    }


def save_data(data: PWRData) -> None:
    data_file.write_text(json.dumps(data, indent=2))


def fetch_from_url(url: str, max_retries: int = 5, delay: int = 1) -> str:
    headers = {"User-Agent": "pwr - paced web reader"}
    for attempt in range(max_retries):
        try:
            print(f"Fetching {url} ...", end="", flush=True)
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            print("DONE")
            return response.text
        except requests.exceptions.HTTPError as e:
            print(f"FAILED {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 4
    print("Max retries reached. Giving up.")
    sys.exit(1)


def get_time() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def print_help() -> None:
    print("Usage: pwr [action]")
    print("\nIf no action is given, cycles through read/fetch/filter in sequence.\n")
    print("Available actions:")
    print("  read   - Read previously selected/enqueued URls")
    print("  fetch  - Retrieve new article titles for review")
    print("  filter - Review article titles and decide which to read")
    print("  status - Print information about current status")


def speedbump(action_name: str, data: PWRData) -> None:
    print(f"About to start pwr {action_name}, last run at: {data['last_'+action_name]}")  # type: ignore
    input(f"Press Enter to continue with pwr {action_name}")


def get_last_action(data: PWRData) -> tuple[str, str]:
    def date_str_for_op(key: str) -> str:
        val = data[key]  # type: ignore
        if val == "Never":
            return "1970-01-01 00:00:00 UTC"
        return val  # type: ignore

    recent_ops = [
        ("read", date_str_for_op("last_read")),
        ("fetch", date_str_for_op("last_fetch")),
        ("filter", date_str_for_op("last_filter")),
    ]
    last_action, last_action_datetime = max(recent_ops, key=lambda x: x[1])
    return (last_action, last_action_datetime)


def count_urls(data: PWRData) -> int:
    return sum(len(source_data["entries"]) for source_data in data["sources"].values())


### Action implementations ###


def do_read() -> None:
    data = load_data()
    last_action, _ = get_last_action(data)
    speedbump("read", data)
    if last_action != "filter":
        print(
            "WARNING: filter is not the most recent action. Did you forget to run it?"
        )
        input("Press Enter to continue anyway, or Ctrl-C to abort")
    urls = extra_urls_for_read.copy()

    for source_data in data["sources"].values():
        for url, _ in source_data["entries"]:
            url = url.split("##")[0]
            if not url.startswith(("http://", "https://")):
                print(f"Skipping url '{url}' as it doesn't have a recognised protocol")
                continue
            urls.append(url)
        source_data["entries"] = []

    print(
        f"Launching browser (in batches of {read_url_batch_size}) for {len(urls)} URLs."
    )

    for url_batch in itertools.batched(urls, read_url_batch_size):
        print(f"Opening batch of URLs with browser {preferred_browser}")
        print(url_batch)
        subprocess.Popen([preferred_browser] + list(url_batch))
        if len(url_batch) == read_url_batch_size:
            input("Press Enter to continue to next batch")

    print("All URLs read, saving changes")
    data["last_read"] = get_time()
    save_data(data)
    print(f"pwr read ended successfully at {data['last_read']}")


class URLCache:
    filename: Path
    data: dict[str, list[str]]
    max_entries: int

    def __init__(self, filename, max_entries=500):
        self.filename = filename
        self.data = dict()
        max_entries = max_entries

        if self.filename.exists():
            print(f"Loading URLs from cache at {self.filename}")
            with open(self.filename, "r") as f:
                self.data = json.load(f)
        else:
            print(f"Created new cache at {self.filename}")
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filename, "w") as f:
                self.data["urls"] = []
                self.filename.write_text(json.dumps(self.data))

    def save(self) -> None:
        self.filename.write_text(json.dumps(self.data))

    def add(self, url: str) -> bool:
        if url not in self.data["urls"]:
            self.data["urls"].append(url)
            self.save()  # This is heavy but performance is not an issue
            return True

        return False


def do_fetch(url_cache: URLCache):
    # [[url,title,source_name]]
    returned_urls: list[list[str]] = []
    total_filtered_extracted = 0
    for source_name, source_info in get_sources().items():
        print(f"Processing source {source_name}")

        func = source_info[0]
        extracted = func(*source_info[1:])

        filtered = [
            [source_name, title, url] for url, title in extracted if url_cache.add(url)
        ]

        returned_urls.extend(filtered)
        total_filtered_extracted += len(filtered)
        print(
            f"Retrieved {len(extracted)} items, {len(filtered)} remain after removing seen items"
        )

    print(
        f"\nA total of {total_filtered_extracted} items were queued up for filtering."
    )

    return pd.DataFrame(returned_urls, columns=["Source", "Title", "URL"])


def get_selected_items(key: str):
    return
    st.write(st.session_state[key])


def streamlit_data_editors(df: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
    """
    Return the dataframes that are generated by streamlit to keep the data alive
    """
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", init_value)

    sources = df_with_selections["Source"].unique()

    # Apply the make_clickable function to the link column
    print(df_with_selections)

    dataframes = []
    for source in sources:
        print(source)
        source_df = df_with_selections[df_with_selections.Source == source].copy()
        source_df = source_df.drop(columns=["Source"])

        st.subheader(source)
        edited_df = st.data_editor(
            source_df,
            key=source,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(required=True),
                "URL": st.column_config.LinkColumn(),
            },
            disabled=df.columns,
            on_change=get_selected_items,
            args=(source,),
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.Select]
        selected_rows.drop("Select", axis=1)
        dataframes.append(selected_rows)

    return dataframes


def do_status() -> None:
    data = load_data()
    last_action, last_action_datetime = get_last_action(data)
    print(f"Last operation was '{last_action}' at {last_action_datetime}")
    print(f"{count_urls(data)} items in entries database")


def do_test_fetcher(name: str, args: list[str]) -> None:
    fetcher_fn = globals().get(f"{name}_fetcher")
    if not fetcher_fn or not callable(fetcher_fn):
        print(f"Error: fetcher function '{name}' not found.")
        sys.exit(1)
    extracted = fetcher_fn(*args)
    print(f"{len(extracted)} URL+title pairs extracted:\n")
    for entry in extracted:
        print(f"URL: {entry[0]}")
        print(f"Title: {entry[1]}")
        print()


### Main ###

if __name__ == "__main__":
    print("Running session")
    data_dir.mkdir(parents=True, exist_ok=True)

    if "data" not in st.session_state:
        st.session_state["data"] = None

    if st.session_state.data is not None:
        data = st.session_state.data
    else:
        data = do_fetch(
            URLCache(Path.home() / ".local" / "share" / "pwr" / "pwr.cache")
        )

    editors = streamlit_data_editors(data)
    st.session_state.data = data

    st.subheader("Your selection:")

    merged_results = pd.DataFrame()
    if editors:
        merged_results = editors[0]
        for df in editors[1:]:
            print(df)
            print(merged_results)
            merged_results = pd.concat([merged_results, df], ignore_index=True)
    else:
        sys.exit(0)

    # merged_results = pd.merge([df for df in editors]) if editors else pd.DataFrame()
    merged_results = merged_results.drop(columns=["Select"])
    # Edit the "Title" column to include the URL as a markdown link
    merged_results["Title"] = merged_results.apply(
        lambda row: f'<a href="{row["URL"]}">{row["Title"]}</a>', axis=1
    )

    # Drop the "URL" column as it's now included in the "Title" column
    merged_results = merged_results.drop(columns=["URL"])

    st.markdown(
        merged_results.style.hide(axis="index").to_html(), unsafe_allow_html=True
    )

    if st.button("Save Data"):
        print("Saving data")
    sys.exit(0)
