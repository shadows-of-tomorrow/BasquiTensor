import os
import time
import requests

FAN_PAINTINGS_URL = "https://www.twoinchbrush.com/images/fanpaintings/"


class TwoInchBrushScraper:

    def __init__(self, dir_out):
        self.dir_out = dir_out
        self.start_index = 7064
        self.end_index = 10000000000000
        self.request_lag = 0.10

    def scrape_images(self):
        for k in range(self.start_index, self.end_index):
            fan_painting_id = f'fanpainting{k}.jpg'
            url = os.path.join(FAN_PAINTINGS_URL, fan_painting_id)
            self._download_image(url, os.path.join(self.dir_out, fan_painting_id))

    def _download_image(self, url, dir):
        r = self._send_get_request(url)
        with open(dir, "wb") as file:
            file.write(r.content)
            file.close()

    def _send_get_request(self, url):
        time.sleep(self.request_lag)
        r = requests.get(url)
        r.close()
        return r


if __name__ == "__main__":
    dir_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'output', 'scraping')
    scraper = TwoInchBrushScraper(dir_out)
    scraper.scrape_images()
