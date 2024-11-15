
# Import Modules and Libraries
# =============================================================================

# Import constants and functions
import Scraper.constants as const
from Scraper.functions import *
# Libraries mainly used for selenium-webscraping:
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import *
from selenium.webdriver import ActionChains
import os
import time

# Libraries mainly used for BeautifulSoup
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from urllib.parse import urljoin

class Casa_scraper(webdriver.Chrome):
# =============================================================================
# We first create a constructur that ensures that the class is initialized with
# all the needed elements
# =============================================================================
    def __init__(self, teardown = False, driver_path =  os.path.dirname(__file__)):
        self.driver_path = driver_path
        os.environ['PATH'] = self.driver_path
        self.teardown = teardown
        self.counter = 0
        self.start_time = time.time()
        
# =============================================================================
# Next we define the options we would like to pass at the beginning to the
# driver
# =============================================================================
        option = webdriver.ChromeOptions()
        option.add_argument('--disable-blink-features=AutomationControlled')
        option.add_argument("window-size=1280, 800")
        option.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64;"\
                            + "x64)AppleWebKit/537.36 (KHTML, like Gecko)"\
                            + "Chrome/74.0.3729.169 Safari/537.36")       

        super(Casa_scraper, self).__init__(options=option)

        # open remote browser in maximized window
        self.maximize_window()

# =============================================================================
# Magic method __exit__ will only run when all other lines of code are done. If
# the user passes teardown = True when he calling the class, the driver will be
# closed at the end.
# =============================================================================
    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = time.time() - self.start_time
        print(f'total time: {total_time}')
        if self.teardown:
            self.quit()

# =============================================================================
# Method will be used to open website main page
# =============================================================================

    def open_main_page(self):
        """
        Method opens the main webpage of www.fotocasa.es
        """
        self.get('https://www.fotocasa.es/es/')
        #self.get('https://www.fotocasa.es/es/comprar/viviendas/barcelona-capital/todas-las-zonas/l')
        
        # Accept cookies if there is any:
        wait = WebDriverWait(self, 10)
        try:
            cookies_element = wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, 'button[data-testid="TcfAccept"]')))
            cookies_element.click()
        except TimeoutException as e:
            print('TimeoutException happened.')
            pass
        finally:
            pass

# =============================================================================
# Method will be used to pass specifications: more if buy or rent and so on
# =============================================================================
    def action(self, action_type = 'buy'):
        
        """
        - action_type: Defines the usage of the website. Accepted values:
            
            - buy: buy an object
            - rent: rent an object
            - new construction: object is a new construction
            - share: share the object with another person
        """
        
        action_type = action_type.lower()
        accepted_values = ['buy', 'rent', 'new construction', 'share']
        class_keys = {
                        'buy': 'buy', 
                        'rent': 'rent', 
                        'new construction': 'FILTER_CONSERVATION_STATUS_NEW_HOME', 
                        'share': 'share'
                        }

        assert action_type in accepted_values
        core_class = f"re-HomeSearchSelector-item re-HomeSearchSelector-item--{class_keys[action_type]}"
        action_element = self.find_element(By.CSS_SELECTOR, f'div[class="{core_class}"]')
        action_element.click()

# =============================================================================
# Method will be used to select the type of object
# =============================================================================
    def select_object(self, object_type = 'housing'):
        
        """
        - object_type: Defines the type of object the user wants to have.
                       Accepted values are:
                           
                           - housing: place to live
                           - new construction: any site which is new built
                           - promotions: any site which is in promotion
                           - commercial: commercial buildings
                           - garage
                           - office
                           - storage-room
                           - land
                           - building: apartment              
        """
        
        object_type = object_type.lower()
        translation_options = {
                                'housing': 'Vivienda', 
                                'new construction': 'Obra nueva', 
                                'promotions': 'Promociones',
                                'commercial': 'Local y nave',
                                'garage': 'Garaje',
                                'office': 'Oficina',
                                'storage-room': 'Trastero',
                                'land': 'Terreno',
                                'building': 'Edificio'
                              }
        
        accepted_values = list(translation_options.keys())
        
        assert object_type in accepted_values
        list_options = self.find_element(By.CSS_SELECTOR, 
                                          'select[title="Seleccione"')

        options = list_options.find_elements(By.TAG_NAME, 'option')
        index = [index for index, word in enumerate(options) 
                 if word.text == translation_options[object_type]]
        options[index[0]].click()
        
# =============================================================================
# Method will be used to select to send address
# =============================================================================
    def set_address(self, **kwargs):
        """
        Function send address to the search engine.
        
        - Valid kwargs:
            - city
            - province
        """
        accepted_values = ['city', 'province', None]
        for key, value in kwargs.items():
            assert key.lower() in accepted_values
            if key.lower() == 'city':
                self.city = value
            elif key.lower() == 'province':
                self.province = value
        
        placeholder = self.find_element(By.CSS_SELECTOR, 'input[placeholder]')
        submit_button = self.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
        placeholder.clear()
        
        query = self.city + ', ' + self.province
        
        
        placeholder.send_keys(query)
        time.sleep(1.5)
        submit_button.click()

        

#%%#  SECTION 2: RETRIVE ADDS INFORMATION - Links
#%%

# =============================================================================
# Method will be used to scroll down through the page in order for the html
# to be displayed completely.
# =============================================================================
    def rolldown_page(self):
        for i in range(0, 2):
            try:
                wait = WebDriverWait(self, 10)
                list_parent = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR,
                                                    'section[class="re-SearchResult"]'))
                    )
                
                scrolling = True
                scroller = 300
                
                while scrolling:
                    try:
                        scroller += 300
                        Y = str(scroller)
                        height0 = self.execute_script('return window.pageYOffset')
                        self.execute_script(f'window.scrollTo(0, {Y})')
                        height1 = self.execute_script('return window.pageYOffset')
                        time.sleep(0.18)
                        if height1 == height0:
                            break
                    except Exception:
                        scrolling = False
            
                break
                        
            except Exception:
                pass

# =============================================================================
# Method will be used to obtain last pate
# =============================================================================
    def last_page(self):
        # find element holding the page buttons:
            
        self.rolldown_page()
        wait = WebDriverWait(self, 15)
        pager = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, 'sui-MoleculePagination'))
                                           )
        buttons = pager.find_elements(By.TAG_NAME, 'li')
            
        # get text inside the buttons
        text = []
        for button in buttons:
            if button.text.isdigit():
                text.append(int(button.text))
        self.lastpage = max(text)
# =============================================================================
# Method will be used to find and go to next page
# =============================================================================
    def next_page(self):
        # find element holding the page buttons
        if self.lastpage > 1:
            while True:
                try:
                    self.execute_script('scrollBy(0, - 2000);')
                    wait = WebDriverWait(self, 15)
                    pager = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'sui-MoleculePagination')))
            
                    buttons = pager.find_elements(By.TAG_NAME, 'li')
                    buttons[-1].click()
                    
                    break
                except Exception:
                    self.close_login(self)
                    pass
 
  
         
   
# =============================================================================
# Method will be used to obtain the links from the page. Here we also use
# BeaufitulSoup to parse the h-refs.
# =============================================================================
    def obtain_hrefs(self):
        # First call method last_page to obtain how many times we need to
        # repeat the process
        self.last_page()
        
        counter = 0
        hrefs = []
        for page_no in range(0, self.lastpage):
            begin_subtime  = time.time()
            
            while True:
                try:
                    if page_no > 0:
                        self.rolldown_page()

                    source = self.page_source
                    soup = BeautifulSoup(source, 'html.parser')
                    articles = soup.find_all('article')

                    for elements in articles:
                        hrefs.append(urljoin(const.base_url, 
                                             elements.find_all('a')[0]['href']))
                    
                    self.counter += add_links_to_database(data = hrefs, city = self.city, db = 'attributes.db', )
                    if page_no < self.lastpage:
                        self.next_page()
                    break
                except Exception:
        
                    self.close_login()
                    pass
        
            end_subtime = time.time()
            total_subtime = end_subtime - begin_subtime
            print(f'page time: {total_subtime}')
# =============================================================================
# Method will be used to close log-in window that sometimes pop out on the
# screen
# =============================================================================
    def close_login(self):
        try:
            wait = WebDriverWait(self, 7)
            close_element = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR,
                                            'button[aria-label="Close Message"]'))
                )
            close_element.click()
        except Exception:
            pass
        
 
#%%
#%%