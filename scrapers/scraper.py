import requests
from bs4 import BeautifulSoup


class UrlScraper:
    """ Extracts all images from a target URL."""
    def __init__(self, name: str, out_dir: str):
        self.name = name
        self.out_dir = out_dir
        self.scraped_images = 1

    def scrape_url(self, url: str):
        image_urls = self.url_to_image_urls(url)
        self.download_images_from_urls(image_urls)

    def download_images_from_urls(self, image_urls):
        for url in image_urls:
            try:
                image_data = requests.get(url).content
                image_dir = self.out_dir / f'{self.name}_{self.scraped_images}.jpg'
                with open(image_dir, 'wb') as handler:
                    handler.write(image_data)
                self.scraped_images += 1
            except ValueError:
                pass

    def url_to_image_urls(self, url: str):
        html = self.url_to_html(url)
        soup = self.html_to_soup(html)
        return self.soup_to_image_urls(soup)

    def soup_to_image_urls(self, soup: BeautifulSoup):
        image_urls = []
        for img in soup.findAll('img'):
            image_urls.append(img.get('src'))
        return image_urls

    def html_to_soup(self, html: str):
        return BeautifulSoup(html, features="html.parser")

    def url_to_html(self, url: str):
        return requests.get(url).text
