""" EDIT THE FILE PATHS ACCORDINGLY """

"""Installations"""

pip install beautifulsoup4
pip install numpy
pip install requests
pip install spacy
pip install trafilatura

"""Imports"""
from bs4 import BeautifulSoup
import json
import numpy as np
import pandas as pd
import glob
import os
import requests
from requests.models import MissingSchema
import spacy
import trafilatura
import time

"""Extracting Text"""

def beautifulsoup_extract_text_fallback(response_content):
    
    # Create the beautifulsoup object:
    soup = BeautifulSoup(response_content, 'html.parser')
    
    # Finding the text:
    text = soup.find_all(text=True)
    
    # Remove unwanted tag elements:
    cleaned_text = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'style',]

    # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
    # is NOT in the blacklist
    for item in text:
        if item.parent.name not in blacklist:
            cleaned_text += '{} '.format(item)
            
    # Remove any tab separation and strip the text:
    cleaned_text = cleaned_text.replace('\t', '')
    return cleaned_text.strip()
    

def extract_text_from_single_web_page(url):
    
    downloaded_url = trafilatura.fetch_url(url)
    
    try:
        a = trafilatura.extract(downloaded_url, output_format='json', with_metadata=False, include_comments = False, include_images = False,                  
              include_tables = False, include_links = False , date_extraction_params={'extensive_search': True, 'original_date': True})
        
    except AttributeError:
        a = trafilatura.extract(downloaded_url, output_format='json', with_metadata=False, include_comments = False, include_images = False,            
                    include_tables = False, include_links = False, date_extraction_params={'extensive_search': True, 'original_date': True})
    
    if a:
        json_output = json.loads(a)
        return json_output['text']
    else:
        try:
            resp = requests.get(url)
            # We will only extract the text from successful requests:
            if resp.status_code == 200:
                return beautifulsoup_extract_text_fallback(resp.content)
            else:
                # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                print('None')
                return np.nan
        # Handling for any URLs that don't have the correct protocol
        except requests.exceptions.RequestException as e:
            print(e)
            return np.nan

"""Label Column Adding"""

real_path = r'path/to/Real'
edited_path = r'path/to/Edited/'
all_files = glob.glob(real_path + "/*.csv")


# get all file names
file_name=[]
for files in all_files:
  file_w_ext = (os.path.basename(files))
  fname, ext = os.path.splitext(file_w_ext)
  file_name.append(fname)



# create new 'Label' column
for i in range(0, len(all_files)):
  csv_file = pd.read_csv(all_files[i],encoding='utf-8')
  csv_file['Label'] = np.nan 
  csv_file.to_csv(edited_path + file_name[i] + '_edited.csv', index=False, encoding='utf-8')
  print(edited_path + file_name[i] + '_edited.csv created')

"""Writing Description in CSV"""

filePath = 'path/to/Edited/' 
editedPath = 'path/to/Crawled/'

all_files = glob.glob(filePath + "/*.csv")


# get all file names
file_name=[]
for files in all_files:
  file_w_ext = (os.path.basename(files))
  fname, ext = os.path.splitext(file_w_ext)
  file_name.append(fname)



unreachableSites = ['http://cumillabarta.com', 'https://www.analysisbd.net', 'https://bit.ly', 'https://www.hasivalobashi.club', 
                    'https://www.bengalbreakingnews.com', 'https://dailymorning24.com', 'https://www.sangbad24x7.com/', 
                    'http://www.naturalhealthtips.us/', 'https://kalerdarpan24.com', 'https://notunalo.press/', 
                    'https://www.timeofkushtia.com/', 'https://somoybd24.info/', 'https://www.sarakhon.com/',
                    'https://www.sheershakhobor.com/']

for j in range(len(all_files)):
  df = pd.read_csv(all_files[j],encoding='utf-8')
  for i in range(len(df)):
    print(i)

    #if Status or Link is of Facebook, dont do anything
    if df.loc[i, 'Type'] == 'Status' or 'https://www.facebook.com' in df.loc[i, 'Link'] :
      print('Status Type')
      print('Continuing')
      continue
      

    elif df.loc[i, 'Type'] == 'Link':
      print('Link Type')
      URL = df.loc[i, 'Link']
 

    else :
      df.drop([i], inplace=True)
      print('Other type')
      print('dropped')
      continue
    
    print(URL)

    if any(x in URL for x in unreachableSites ):
      df.drop([i], inplace = True )
      print('dropped as unreachable')
      continue


    try:
      text = extract_text_from_single_web_page(url=URL)
    except KeyboardInterrupt as e:
      print('Key pressed')
      df.drop([i], inplace = True )
      print('dropped as key pressed')
      continue
      
    

    #  if the url is unreachable, drop it
    if text is np.nan:
      df.drop([i], inplace = True )
      print('dropped')
      continue

    # replace Description with text extracted from the link
    df.replace(to_replace = df.loc[i, 'Description'], 
                 value = text, 
                  inplace = True)
    print("Description Updated")
    

  df.to_csv(editedPath + file_name[j] +  'crawled.csv', index=False, encoding='utf-8')
  print(editedPath + file_name[j] +  'crawled.csv created')

