import os
import time
import json
import requests

MET_API_URL = "https://collectionapi.metmuseum.org/public/collection/v1/"


class MetAPIScraper:

    def __init__(self, dir_out):
        self.base_url = self._construct_base_url()
        self.object_id_key = 'objectIDs'
        self.image_key = 'primaryImage'
        self.image_check = 'hasImage'
        self.search_endpoint = 'search'
        self.object_endpoint = 'objects'
        self.image_dir = dir_out
        self.request_lag = 0.00

    def scrape_images(self, **kwargs):
        object_ids = self._find_object_ids(**kwargs)
        for object_id in object_ids:
            r = self._send_object_request(object_id)
            img_url = self._get_image_url(r)
            self._download_image(img_url, self.image_dir)

    def _download_image(self, url, dir):
        name = url.split('/')[-1]
        r = self._send_get_request(url)
        with open(os.path.join(dir, name), "wb") as file:
            file.write(r.content)
            file.close()

    def _get_image_url(self, r):
        img_url = self._parse_requests([r], [self.image_key])[self.image_key]
        return img_url[0]

    def _get_objects(self, **kwargs):
        obs = []
        ob_ids = self._find_object_ids(**kwargs)
        for ob_id in ob_ids:
            ob = self._send_object_request(ob_id)
            obs.append(ob)
        return obs

    def _find_object_ids(self, **kwargs):
        r = self._send_search_request(**kwargs)
        object_ids = self._parse_request(r, [self.object_id_key])
        return object_ids.get(self.object_id_key)

    def _construct_search_string(self, **kwargs):
        url = self.base_url + self.search_endpoint + '?'
        for key, value in kwargs.items():
            url = url + key + '=' + value + '&'
        url = url[:-1]
        return url

    def _send_object_request(self, object_id):
        url = self.base_url + self.object_endpoint + f'/{object_id}'
        return self._send_get_request(url)

    def _send_search_request(self, **kwargs):
        url = self._construct_search_string(**kwargs)
        return self._send_get_request(url)

    def _parse_requests(self, rs, keys):
        rs_parsed = {k: [] for k in keys}
        for r in rs:
            r_parsed = self._parse_request(r, keys)
            rs_parsed = self._add_dict(rs_parsed, r_parsed)
        return rs_parsed

    @staticmethod
    def _parse_request(r, keys):
        json_dict = json.loads(r.text)
        return {k: json_dict[k] for k in json_dict.keys() if k in keys}

    @staticmethod
    def _construct_base_url():
        return MET_API_URL

    def _send_get_request(self, url):
        time.sleep(self.request_lag)
        r = requests.get(url)
        r.close()
        return r

    @staticmethod
    def _add_dict(dict_old, dict_new):
        for key, value in dict_new.items():
            dict_old[key].append(value)
        return dict_old


if __name__ == "__main__":
    dir_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'output', 'scraping')
    scraper = MetAPIScraper(dir_out)
    scraper.scrape_images(q="Portrait", medium="Paintings", isOnView="true")
