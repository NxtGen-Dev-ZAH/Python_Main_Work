import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin


def download_images(query, num_images=100):
    headers = {"User-Agent": "Mozilla/5.0"}
    url_template = f"https://www.pixabay.com/images/search/{query}"
    downloaded = 0
    start = 0

    if not os.path.exists(query):
        os.makedirs(query)

    while downloaded < num_images:
        url = url_template.format(start)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        images = soup.find_all("img")

        for i, img in enumerate(images):
            img_url = img.get("src")
            if img_url:
                # Handle relative URLs
                if not img_url.startswith(("http://", "https://")):
                    img_url = urljoin(url, img_url)
                try:
                    img_data = requests.get(img_url).content
                    with open(f"{query}/{query}_{downloaded + i}.jpg", "wb") as handler:
                        handler.write(img_data)
                    downloaded += 1
                    if downloaded >= num_images:
                        break
                except requests.exceptions.RequestException as e:
                    print(f"Could not download {img_url}: {e}")

        start += len(images)  # Move to the next set of images

        if len(images) == 0:
            print("No more images found.")
            break

    print(f"Downloaded {downloaded} images of {query}")


download_images("motorcycles", num_images=100)
