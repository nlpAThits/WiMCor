from bs4 import BeautifulSoup
import sys
import requests
from services import get_tic, compute_elapsed_time
from multiprocessing import Pool as Threadpool
from multiprocessing import cpu_count

session = requests.Session()
url = "https://en.wikipedia.org/w/api.php"

def get_ancestors(current):
    ancestors = []
    while current.name != '[document]':
        ancestors.append(current.parent.name)
        current = current.parent
    return ancestors

def parse_disambiguation_page(title):

    params = {
            "action": "parse",
            "page": title,
            "format": "json"}

    try:
        data = session.get(url=url, params=params).json()
        content = data["parse"]["text"]["*"]
    except Exception: # Page not found
        return []

    soup = BeautifulSoup(content, 'html.parser')    
    wikilinks = []

    try:
        '''
            Pick the first link in the introduction paragraph in a disambiguation page.
        '''
        hyperlink = soup.find('p').find('a')
        if '/wiki/' in hyperlink['href']:
            wikilinks.append(hyperlink.text)
            # print('\t{}'.format(hyperlink.text))
    except Exception: # No paragraph.
        pass

    list_items = soup.find_all('li')
    for list_item in list_items:
        try:
            hyperlink = list_item.find('a')
            '''
                Pick a hyperlink from each list item.

                So heuristics to remove noise:
                    1. There is no string before the hyperlink.
                    2. The hyperlink is an active wikilink and not a dead link.
                    3. The hyperlink does not point to another disambiguation page.
                    4. The hyperlink is not under 'See also' heading.
                    5. The hyperlink is not in a table.
            '''
            if  list_item.text.startswith(hyperlink.text):
                if '/wiki/' in hyperlink['href']:
                    if '(disambiguation)' not in hyperlink['href']:
                        if not (list_item.parent.find_previous_sibling('h2') and 'See also' in list_item.parent.find_previous_sibling('h2').text):
                            if 'table' not in get_ancestors(hyperlink):
                                wikilinks.append(hyperlink.text)
                                # print('\t{}'.format(hyperlink.text))
        except Exception: # No hyperlink in list item.
            continue

    return wikilinks

if __name__=="__main__":   
    tic = get_tic()

    with open('/home/mathewkn/metonymy-resolution/harvest-data/disambiguation-pages/apiwikipedia/disambiguation_page_titles_uniqsort') as fp:
        disambiguation_page_titles = fp.readlines()

    disambiguation_page_titles = [title.strip() for title in disambiguation_page_titles] 

    for title in disambiguation_page_titles:
        parse_disambiguation_page(title)

    compute_elapsed_time(tic)
