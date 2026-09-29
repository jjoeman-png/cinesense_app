"""On-demand, cached movie metadata from CC0 Wikidata records."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

import streamlit as st


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_USER_AGENT = "CineSenseCS315/1.0 (educational movie analytics project)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.blocked_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "sup"}:
            self.blocked_depth += 1
        elif tag in {"p", "li", "h2", "h3", "h4", "br"} and not self.blocked_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "sup"} and self.blocked_depth:
            self.blocked_depth -= 1

    def handle_data(self, data):
        if not self.blocked_depth:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


def _fetch_wikipedia_plot(article_url: str) -> dict[str, str]:
    if not article_url.startswith("https://en.wikipedia.org/wiki/"):
        return {}
    title = urllib.parse.unquote(article_url.rsplit("/", 1)[-1]).replace("_", " ")
    headers = {"User-Agent": WIKIDATA_USER_AGENT, "Accept": "application/json"}
    try:
        section_query = urllib.parse.urlencode({"action": "parse", "page": title, "prop": "sections", "format": "json"})
        section_request = urllib.request.Request(f"{WIKIPEDIA_API}?{section_query}", headers=headers)
        sections = json.load(urllib.request.urlopen(section_request, timeout=12)).get("parse", {}).get("sections", [])
        plot_section = next((
            section for section in sections
            if section.get("line", "").strip().casefold() in {"plot", "plot summary", "synopsis"}
        ), None)
        if plot_section is None:
            return {}
        text_query = urllib.parse.urlencode({
            "action": "parse", "page": title, "section": plot_section["index"], "prop": "text", "format": "json",
        })
        text_request = urllib.request.Request(f"{WIKIPEDIA_API}?{text_query}", headers=headers)
        html_text = json.load(urllib.request.urlopen(text_request, timeout=12)).get("parse", {}).get("text", {}).get("*", "")
        extractor = _HTMLText()
        extractor.feed(html_text)
        plot = " ".join(" ".join(extractor.parts).split())
        return {"wikipedia_plot_excerpt": plot[:2400], "wikipedia_url": article_url} if plot else {}
    except Exception as error:
        print(f"[Wikipedia] plot lookup failed: {error!r}")
        return {}


def _api_json(endpoint: str, params: dict[str, str]) -> dict:
    request_url = endpoint + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": WIKIDATA_USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.load(response)


def _normalized_article_title(title: str) -> str:
    title = re.sub(r"\s*\(\d{4}\)$", "", title).casefold()
    inverted_article = re.match(r"^(.*),\s*(the|a|an)$", title)
    if inverted_article:
        title = f"{inverted_article.group(2)} {inverted_article.group(1)}"
    title = re.sub(r"^(the|a|an)\s+", "", title)
    return re.sub(r"[^a-z0-9]+", "", title)


def _entity_claim_values(entity: dict, property_id: str) -> list[dict]:
    values = []
    for claim in entity.get("claims", {}).get(property_id, []):
        snak = claim.get("mainsnak", {})
        if snak.get("snaktype") == "value":
            value = snak.get("datavalue", {}).get("value")
            if value is not None:
                values.append(value)
    return values


def _metadata_from_wikipedia(imdb_id: str, movie_title: str, year: int | None) -> dict[str, object]:
    """Fallback to Wikipedia search/page APIs and validate the linked IMDb ID."""
    queries = [f'insource:"{imdb_id}"']
    title_query = f'"{movie_title}" {year} film' if year else f'"{movie_title}" film'
    queries.append(title_query)
    article_title = None

    try:
        for query in queries:
            search = _api_json(WIKIPEDIA_API, {
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": "10",
            })
            candidates = search.get("query", {}).get("search", [])
            exact = [
                item["title"] for item in candidates
                if _normalized_article_title(item["title"]) == _normalized_article_title(movie_title)
            ]
            if exact:
                article_title = exact[0]
                break
    except Exception as error:
        print(f"[Wikipedia] article search failed: {error!r}")
        return {}

    if not article_title:
        return {}

    try:
        page_data = _api_json(WIKIPEDIA_API, {
            "action": "query", "prop": "pageprops|pageimages", "piprop": "thumbnail",
            "pithumbsize": "640", "titles": article_title, "format": "json",
        })
        pages = page_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        item_id = page.get("pageprops", {}).get("wikibase_item", "")
        if not item_id:
            return {}

        entity_data = _api_json(f"https://www.wikidata.org/wiki/Special:EntityData/{item_id}.json", {})
        entity = entity_data.get("entities", {}).get(item_id, {})
        external_ids = {
            str(value).removeprefix("tt").zfill(7)
            for value in _entity_claim_values(entity, "P345")
        }
        if imdb_id.removeprefix("tt").zfill(7) not in external_ids:
            return {}

        person_ids = list(dict.fromkeys(
            value["id"]
            for property_id in ("P57", "P161")
            for value in _entity_claim_values(entity, property_id)
            if isinstance(value, dict) and value.get("id")
        ))
        labels = {}
        for offset in range(0, len(person_ids), 50):
            label_data = _api_json(WIKIDATA_API, {
                "action": "wbgetentities", "ids": "|".join(person_ids[offset:offset + 50]),
                "props": "labels", "languages": "en", "format": "json",
            })
            for identifier, person in label_data.get("entities", {}).items():
                label = person.get("labels", {}).get("en", {}).get("value")
                if label:
                    labels[identifier] = label

        directors = [labels[value["id"]] for value in _entity_claim_values(entity, "P57") if value.get("id") in labels][:5]
        cast = [labels[value["id"]] for value in _entity_claim_values(entity, "P161") if value.get("id") in labels][:10]

        runtime = None
        runtime_values = _entity_claim_values(entity, "P2047")
        if runtime_values:
            runtime = float(runtime_values[0]["amount"].lstrip("+"))
            if runtime_values[0].get("unit", "").endswith("Q11574"):
                runtime /= 60
            runtime = round(runtime, 1)

        image_values = _entity_claim_values(entity, "P18")
        image_name = str(image_values[0]) if image_values else ""
        image_url = (
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            + urllib.parse.quote(image_name.replace("_", " "))
            + "?width=640"
            if image_name else ""
        )
        image_page = (
            "https://commons.wikimedia.org/wiki/File:"
            + urllib.parse.quote(image_name.replace(" ", "_"), safe="()!,._-")
            if image_name else ""
        )
        article_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(article_title.replace(" ", "_"), safe="()!._-:')")
        result = {
            "wikidata_id": item_id,
            "wikidata_url": f"https://www.wikidata.org/wiki/{item_id}",
            "description": entity.get("descriptions", {}).get("en", {}).get("value", ""),
            "directors": directors,
            "cast": cast,
            "runtime_minutes": runtime,
            "image_url": image_url,
            "image_page_url": image_page,
            "image_filename": image_name,
        }
        result.update(_fetch_wikipedia_plot(article_url))
        return result
    except Exception as error:
        print(f"[Wikidata] entity API fallback failed: {error!r}")
        return {}


def fetch_wikidata_metadata(imdb_id: str, movie_title: str = "", year: int | None = None) -> dict[str, object]:
    """Look up optional movie details by the IMDb ID supplied in MovieLens links."""
    raw_id = str(imdb_id).strip().lower()
    if raw_id.startswith("tt"):
        raw_id = raw_id[2:]
    if not raw_id.isdigit():
        return {"error": "This title has no valid IMDb cross-reference in the MovieLens links file."}
    normalized_id = f"tt{raw_id.zfill(7)}"

    query = f'''SELECT ?item ?description ?directorLabel ?castLabel ?runtimeAmount ?runtimeUnit ?image ?wikipediaArticle WHERE {{
      ?item wdt:P345 "{normalized_id}".
      OPTIONAL {{ ?item schema:description ?description. FILTER(LANG(?description) = "en") }}
      OPTIONAL {{ ?item wdt:P57 ?director. }}
      OPTIONAL {{ ?item wdt:P161 ?cast. }}
      OPTIONAL {{
        ?item p:P2047/psv:P2047 ?runtimeValue.
        ?runtimeValue wikibase:quantityAmount ?runtimeAmount;
                      wikibase:quantityUnit ?runtimeUnit.
      }}
      OPTIONAL {{ ?item wdt:P18 ?image. }}
    OPTIONAL {{ ?wikipediaArticle schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 100'''
    request_url = WIKIDATA_ENDPOINT + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
    request = urllib.request.Request(
        request_url,
        headers={
            "User-Agent": WIKIDATA_USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            rows = json.load(response)["results"]["bindings"]
    except Exception as error:
        print(f"[Wikidata] metadata lookup failed: {error!r}")
        fallback = _metadata_from_wikipedia(normalized_id, movie_title, year) if movie_title else {}
        if fallback:
            return fallback
        return {"error": "Wikidata is rate-limiting metadata requests. The MovieLens title and ratings still work; try again later."}

    if not rows:
        fallback = _metadata_from_wikipedia(normalized_id, movie_title, year) if movie_title else {}
        if fallback:
            return fallback
        return {"error": "No matching Wikidata item was found for this MovieLens IMDb ID."}

    def values(field: str) -> list[str]:
        return list(dict.fromkeys(row[field]["value"] for row in rows if field in row))

    item_uri = values("item")
    descriptions = values("description")
    directors = [value for value in values("directorLabel") if not re.fullmatch(r"Q\d+", value)][:5]
    cast = [value for value in values("castLabel") if not re.fullmatch(r"Q\d+", value)][:10]
    runtime_rows = [row for row in rows if "runtimeAmount" in row and "runtimeUnit" in row]
    runtime_values = []
    for row in runtime_rows:
        amount = float(row["runtimeAmount"]["value"])
        unit_uri = row["runtimeUnit"]["value"]
        if unit_uri.endswith("Q11574"):
            amount /= 60
        runtime_values.append(round(amount, 1))

    images = values("image")
    image_name = images[0].rsplit("/", 1)[-1] if images else ""
    image_url = (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        + urllib.parse.quote(image_name.replace("_", " "))
        + "?width=640"
        if image_name else ""
    )
    image_page = (
        "https://commons.wikimedia.org/wiki/File:"
        + urllib.parse.quote(image_name.replace(" ", "_"), safe="()!,._-")
        if image_name else ""
    )
    item_id = item_uri[0].rsplit("/", 1)[-1] if item_uri else ""
    wikipedia_articles = values("wikipediaArticle")
    result = {
        "wikidata_id": item_id,
        "wikidata_url": f"https://www.wikidata.org/wiki/{item_id}" if item_id else "",
        "description": descriptions[0] if descriptions else "",
        "directors": directors[:5],
        "cast": cast[:10],
        "runtime_minutes": runtime_values[0] if runtime_values else None,
        "image_url": image_url,
        "image_page_url": image_page,
        "image_filename": image_name,
    }
    if wikipedia_articles:
        result.update(_fetch_wikipedia_plot(wikipedia_articles[0]))
    return result