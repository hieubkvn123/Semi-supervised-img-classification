import os
import sys
import cv2
import time
import tqdm
import requests
import traceback
import numpy as np
import urllib.request as urllib
from base64 import decodebytes

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--driver', required=True, type=str, help='Path to webdriver')
parser.add_argument('--query', required=True, type=str, help='Search query to look up')
parser.add_argument('--dir', required=True, type=str, help='Base directory to store scraped images')
args = vars(parser.parse_args())

options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(args['driver'], options=options)

def go_to(driver, url):
	driver.get(url)

def enter_input(driver, input_xpath, text):
	element = driver.find_element_by_xpath(input_xpath)
	element.send_keys(text)
	element.send_keys(Keys.ENTER)

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

def find_and_download(driver, class_name, folder='images', prefix='img', num_to_download=100):
	if(not os.path.exists(folder)):
		print('[INFO] Creating image folder ... ')
		os.makedirs(folder, exist_ok=True)

	num_downloaded = 0
	downloaded = set()
	SCROLL_PAUSE_TIME = 0.5

	while(num_downloaded < num_to_download):
		img_elements = driver.find_elements_by_class_name(img_class)

		for elem in img_elements:
			if(elem in downloaded): 
				continue 

			img_src = elem.get_attribute('src')
			if(img_src is None):
				continue

			if(num_downloaded >= num_to_download):
				break

			try:
				print(f'    --> Downloading image #{num_downloaded+1}')
				if(img_src.startswith('data:image/jpeg;base64')):
					img_string = elem.get_attribute('src').split(',')[-1]

					base64_img_bytes = img_string.encode('utf-8')
					with open(f'{folder}/{prefix}_{time.time()}.png', 'wb') as f:
						decoded_image_data = decodebytes(base64_img_bytes)
						f.write(decoded_image_data)
				elif(img_src.startswith('https')):
					img = url_to_image(img_src)
					cv2.imwrite(f'{folder}/{prefix}_{time.time()}.png', img)

				num_downloaded += 1
				downloaded.add(elem)
			except:
				print(f'    --> Error downloading image ')
				traceback.print_exc(file=sys.stdout)
				# driver.quit()

		# Scroll down to bottom
		driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

		# Wait to load page
		time.sleep(SCROLL_PAUSE_TIME)

if __name__ == '__main__':
	input_xpath = '/html/body/div[2]/div[2]/div/form/div[1]/div[1]/div[1]/div/div[2]/input'
	img_class = 'rg_i'

	go_to(driver, 'https://www.google.com/imghp?hl=EN')
	enter_input(driver, input_xpath, args['query'])
	find_and_download(driver, img_class, folder=args['dir'], prefix=args['query'])

	driver.quit()
