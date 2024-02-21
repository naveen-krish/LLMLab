import streamlit as st
import requests
from bs4 import BeautifulSoup

def scrape_wiki_to_txt(page_title):
    url = f"https://en.wikipedia.org/wiki/{page_title}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find(id="mw-content-text").get_text()
    return content

st.title('Wikipedia Page Scraper')

page_title = st.text_input('Enter Wikipedia Page Title', '')

#if st.button('Test Input'):
  #  st.write(f"You entered: {page_title}")
    
     #Scrape and display content if a page title is entered and button is clicked
if page_title:
    if st.button('Scrape Page'):
        content = scrape_wiki_to_txt(page_title.replace(' ', '_'))
        st.text_area('Scraped Content', content, height=300)
        st.download_button('Download Content', content, f"{page_title.replace(' ', '_')}.txt")
