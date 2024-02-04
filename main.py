# Importing necessary modules
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
 
# WebDriver Chrome
driver = webdriver.Safari()
 
# Target URL
driver.get("https://www.buyersproducts.com/catalog/snowdogg-9")
# To load entire webpage
time.sleep(5)
 
# Printing the whole body text
print(driver.find_element(By.XPATH, "/html/body").text)
 
# Closing the driver
driver.close()