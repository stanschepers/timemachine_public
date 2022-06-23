from bs4 import BeautifulSoup
import requests
import re, os
import random

import pandas as pd

from tqdm import tqdm


def get_page_url(page_nr=0):
    return f"https://www.shorpy.com/node?page={page_nr}"


def resolve_year(title):
    pattern = r"(?P<YEAR>\d\d\d\d)"
    m = re.findall(pattern, title)
    if len(m) == 0:
        return None
    return m[-1]


def download_image(url, filename):
    try:
        if os.path.exists(filename):
            return True
        response = requests.get(url)
        with open(filename, "wb+") as f:
            f.write(response.content)
        return True
    except:
        return False


def download_pages():
    page_nr = 0
    img_counter = 0
    try:
        data = []
        except_counter = 0

        for page_nr in tqdm(range(0, 1044)):
            page_url = get_page_url(page_nr)
            page = requests.get(page_url)
            soup = BeautifulSoup(page.text, "lxml")
            for content in soup.select("div.node > div.content"):
                try:
                    img_counter += 1
                    img = content.select("a > img")[0]
                    p = content.select("p")[0]
                    src = img["src"]
                    title = img["title"]
                    year = int(resolve_year(title))
                    text = p.text[:-16]
                    filename = f"images/{img_counter}_{year}.jpg"
                    downloaded = download_image(src, filename)
                    data.append((year, src, text, filename))
                except Exception as e:
                    except_counter += 1

    except Exception as e:
        print(e)
        print("Stopped at", page_nr)

    finally:
        df = pd.DataFrame(data, columns=["year", "src", "text", "filename"])
        df.to_csv("info.csv", index=False)
        return df


if __name__ == "__main__":
    download_pages()