import os
import time
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options

MIN_BYTE_SIZE = 10000
WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
FINISHED_COLOR = '\033[92m'
END_COLOR = '\033[0m'


class DomainScraper:
    """
    Crawls through a domain and extracts all the images from anchor tags.
    """
    def __init__(self, root_url):
        self.root_url = root_url
        self.data_folder = self._folder_from_root_url()
        self.min_byte_size = MIN_BYTE_SIZE
        self.scraped_urls = []
        self.queued_urls = []
        self._create_data_folder()
        self.request_delay = 0.10
        self.browser = self._construct_browser()

    def run(self):
        self._scrape_url(self.root_url)
        self._process_queue()
        self.browser.close()
        print(f"{FINISHED_COLOR}Thread <{self.root_url}> has finished scraping.")

    def _construct_browser(self):
        options = Options()
        options.add_argument("--headless")
        driver_path = os.path.join(os.path.dirname(__file__), 'webdriver', 'chromedriver.exe')
        return webdriver.Chrome(executable_path=driver_path, options=options)

    def _folder_from_root_url(self):
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', self.root_url[8:])

    def _process_queue(self):
        while len(self.queued_urls) > 0:
            try:
                next_url = self.queued_urls.pop()
                self._scrape_url(next_url)
            except Exception as e:
                print(f"{ERROR_COLOR}Thread <{self.root_url}> fatal error: {e}{END_COLOR}")

    def _scrape_url(self, url):
        print(f"Thread <{self.root_url}> is scraping: {url}")
        html_text = self._get_html_text(url)
        self._scrape_imgs_from_html(html_text)
        self.scraped_urls.append(url)
        candidate_urls = self._extract_href_urls(html_text)
        self._queue_candidate_urls(candidate_urls)

    def _create_data_folder(self):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def _queue_candidate_urls(self, candidate_urls):
        valid_urls = self._clean_candidate_urls(candidate_urls)
        self.queued_urls = valid_urls + self.queued_urls

    def _clean_candidate_urls(self, urls):
        valid_urls = self._remove_duplicate_urls(urls)
        valid_urls = self._remove_cached_urls(valid_urls)
        valid_urls = self._remove_out_of_domain_urls(valid_urls)
        return valid_urls

    def _remove_out_of_domain_urls(self, urls):
        valid_urls = [url for url in urls if self.root_url in url]
        return valid_urls

    def _remove_cached_urls(self, urls):
        valid_urls = list(set(urls) - set(self.scraped_urls + self.queued_urls))
        return valid_urls

    def _remove_duplicate_urls(self, urls):
        return list(set(urls))

    def _update_root_url(self, root_url):
        self.root_url = root_url

    def _scrape_imgs_from_html(self, html_text):
        src_urls = self._extract_src_urls(html_text)
        self._download_from_src_urls(src_urls)

    def _download_from_src_urls(self, src_urls):
        for src_url in src_urls:
            try:
                content = requests.get(src_url).content
                self._write_content_to_disk(content)
            except Exception as e:
                print(f"{WARNING_COLOR}Thread <{self.root_url}> failed: {e}{END_COLOR}")

    def _write_content_to_disk(self, content):
        if len(content) > self.min_byte_size:
            file_name = f"{self._generate_timestamp()}.png"
            write_dir = os.path.join(self.data_folder, file_name)
            with open(write_dir, "wb") as f:
                f.write(content)

    def _generate_timestamp(self):
        return str(int(time.time()*1000))

    def _extract_src_urls(self, html_text):
        soup = bs(html_text, features="html.parser")
        src_urls = []
        for img_tag in soup.find_all('img'):
            if self._validate_img_tag(img_tag):
                src_url = self._process_img_tag(img_tag)
                src_urls.append(src_url)
        return src_urls

    def _process_img_tag(self, img_tag):
        src_url = img_tag['src']
        src_url = self._clean_src_url(src_url)
        return src_url

    def _validate_img_tag(self, img_tag):
        if 'src' not in img_tag.attrs.keys():
            return False
        return True

    def _clean_src_url(self, src_url):
        src_url = self._adjust_relative_url(src_url)
        src_url = self._remove_styling_params(src_url)
        return src_url

    def _is_url_relative(self, url):
        if url[0] == "/":
            return True
        else:
            return False

    def _remove_styling_params(self, url):
        return url.split("?")[0]

    def _adjust_relative_url(self, url):
        if self._is_url_relative(url):
            url = self.root_url + url
        return url

    def _extract_href_urls(self, html_text):
        soup = bs(html_text, features="html.parser")
        href_urls = []
        for anc_tag in soup.find_all('a'):
            if self._validate_anchor_tag(anc_tag):
                href_url = self._process_anchor_tag(anc_tag)
                href_urls.append(href_url)
        return href_urls

    def _process_anchor_tag(self, anc_tag):
        href_url = anc_tag['href']
        href_url = self._clean_href_url(href_url)
        return href_url

    def _validate_anchor_tag(self, anc_tag):
        if 'href' not in anc_tag.attrs.keys():
            return False
        else:
            href = anc_tag['href']
            if len(href) == 0:
                return False
            if href[0] == "#":
                return False
            if ".pdf" in href:
                return False
        return True

    def _clean_href_url(self, href_url):
        href_url = self._adjust_relative_url(href_url)
        return href_url

    def _get_page_source(self, url):
        time.sleep(self.request_delay)
        self.browser.get(url)
        WebDriverWait(self.browser, 500)
        return self.browser.page_source

    def _get_html_text(self, url):
        try:
            html_text = self._get_page_source(url)
            return html_text
        except Exception as e:
            print(f"{WARNING_COLOR}Thread <{self.root_url}> failed: {e}{END_COLOR}")
            return ""
