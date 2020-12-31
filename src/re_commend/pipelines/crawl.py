from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from kedro.pipeline import Pipeline, node
from typing import Callable, Dict, Any


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=list_brands_from_breed,
                inputs=[],
                outputs="brands"
            ),
            node(
                func=crawl_brands,
                inputs=["brands"],
                outputs="brand_details"
            )
        ]
    )


def list_brands_from_breed() -> pd.DataFrame:
    breed_url = request.urlopen("http://re-comme-nd.jp/breed/")

    soup = BeautifulSoup(breed_url)
    return pd.DataFrame([
        {"breed": breed, "brand": name, "url": link}
        for breed, name, link in iter_breed(soup)
    ]).reset_index()


def iter_breed(soup: BeautifulSoup):
    for header in soup.find("section", {"id": "archive"}).find_all("h2"):
        breed = header.text
        for item in header.find_next_sibling("ul").find_all("li"):         
            a = item.find("a")
            # print(a)
            name = item.find("div", {"class": "txt"}).find("h3").text
            link = a.attrs["href"]
            yield (breed, name, link)


def crawl_brands(brands: pd.DataFrame) -> Dict[str, Any]:
    # paritioned by index
    return {
        f"{index:03d}": fetch_rice(BeautifulSoup(request.urlopen(url)))
        for index, url in zip(brands["index"], brands["url"])
    }


def fetch_rice(soup: BeautifulSoup):
    sleep(1)
    review_text = ""
    description = ""
    characters = ""
    try:
        review = soup.select("#rice #review")[0]
        review_text = review.select("div.text p")[0].text
        description = review.find_next_sibling("div").find("p").text
        characters = soup.select("#rice div.text table")[0].find_all("td")[2].text
    except Exception as e:
        print(e)
    return {"review": review_text, "description": description, "characters": characters}
