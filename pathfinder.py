import argparse
import wikipediaapi
from bs4 import BeautifulSoup
from itertools import combinations, product
import bz2
import re
import sys
import os
import xml.sax
import subprocess
import mwparserfromhell
from multiprocessing import Pool as Threadpool
from multiprocessing import cpu_count
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from services import get_tic, compute_elapsed_time
from wiki import get
from sojourner import matches_association_pair, matches_association_element, get_remote_links, get_remote_summary
from scrapy import parse_disambiguation_page

'''

    This is built using the blog article:
        https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c

    A list of infoboxes is available at:
        https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes

        Template:Infobox company (Transclusion count: 52,291)
        Template:Infobox military conflict (Transclusion count: 12,889)
        Template:Infobox university (Transclusion count: 24,284 )
        Template:Infobox organization (Transclusion count: 14,497)

        Template:Infobox school (Transclusion count: 36,914)
        Template:Infobox legislature (Transclusion count: 1,665)
        Template:Infobox football club (Transclusion count: 18,206)
        ...

        My observation is DBpedia is better than Wikipedia category graph or Wikipedia infoboxes.
    '''

parser = argparse.ArgumentParser()
parser.add_argument('--disamb_file', default='disambiguation_page_titles_uniqsort_unleash', help="List of disambiguation pages")
parser.add_argument('-vehicles', nargs='+', help="Vehicles")
parser.add_argument('-targets', nargs='+', help="Targets")
args = parser.parse_args()
print(args)

datafile = '/home/mathewkn/metonymy-resolution/harvest-data/disambiguation-pages/apiwikipedia/20190901/enwiki-20190901-pages-articles-multistream.xml.bz2'
indexfile = '/home/mathewkn/metonymy-resolution/harvest-data/disambiguation-pages/apiwikipedia/20190901/index'

def get_summary(title):
    try:
        title, text = get(datafile, indexfile, title, raw=True)
        wikicode = mwparserfromhell.parse(text)
        summary = wikicode.get_sections()[0].strip_code()
    except Exception:
        print('Exception caught in get_summary(): {}'.format(title))
        summary = get_remote_summary(title)

    return summary

def get_links(title):
    try:
        title, text = get(datafile, indexfile, title, raw=True)
        wikicode = mwparserfromhell.parse(text)
        links = [x.title.strip_code().strip() for x in wikicode.filter_wikilinks()]
    except Exception:
        print('Exception caught in get_links(): {}'.format(title))
        links = get_remote_links(title)

    return links

def get_associations():
    metonymic_associations = []

    for (a, b) in product(args.vehicles, args.targets):
        metonymic_associations.append((a, b))

    '''
    PopulatedPlace
    Person100007846

    EducationalInstitution, EducationalInstitution108276342
    Artifact100021939, Product104007894, Creation103129123, Structure104341686, Building102913152
    Company108058098, Association108049401
    Scandal107223811, Event100029378
    '''
    return metonymic_associations

def dumps_extractor(disambiguation_page_title):
    # strip the line of whitespaces
    disambiguation_page_title = disambiguation_page_title.strip()

    print('Processing disambiguation page: {}'.format(disambiguation_page_title))

    metonymic_associations = get_associations()

    links = parse_disambiguation_page(disambiguation_page_title)

    # Filter links
    elements = set([item for sublist in metonymic_associations for item in sublist])
    wikilinks = [link for link in links if matches_association_element(link, elements)]

    # get instances matching an association
    for (foo, bar) in combinations(wikilinks, 2):
        '''
            'University of Oxford' in en_wiki.page('Oxford').wikilinks.keys()
            'University of Oxford' in en_wiki.page('Oxford').backwikilinks.keys()
        '''
        try:
            # if re.search(foo, get_summary(bar), re.IGNORECASE):
                # if re.search(bar, get_summary(foo), re.IGNORECASE):
                    if foo in get_links(bar):
                        if bar in get_links(foo):
                            if matches_association_pair(foo, bar, metonymic_associations):
                                print('Hit {}: (<{}>, <{}>)'.format(disambiguation_page_title, foo, bar))
                            elif matches_association_pair(bar, foo, metonymic_associations):
                                print('Hit {}: (<{}>, <{}>)'.format(disambiguation_page_title, bar, foo))
        except Exception:
            print('Exception caught in dumps_extractor():{}'.format(disambiguation_page_title))

    print('Done with disambiguation page: {}'.format(disambiguation_page_title))
    return

if __name__=="__main__":
    tic = get_tic()

    with open('/home/mathewkn/metonymy-resolution/harvest-data/disambiguation-pages/apiwikipedia/{}'.format(args.disamb_file)) as fp:
        disambiguation_page_titles = fp.readlines()

    '''
    for disambiguation_page_title in disambiguation_page_titles[400:]:
        dumps_extractor(disambiguation_page_title)
    '''

    '''
    # Create a threadpool for reading in files
    threadpool = Threadpool(cpu_count())

    # Read in the files as a list of lists
    results = threadpool.map(dumps_extractor, disambiguation_page_titles)

    threadpool.close()
    threadpool.join()
    '''

    # https://stackoverflow.com/questions/44402085/multiprocessing-map-over-list-killing-processes-that-stall-above-timeout-limi/44404854
    with ProcessPool(cpu_count()) as pool:
        future = pool.map(dumps_extractor, disambiguation_page_titles, timeout=1800) # testing with HALF AN HOUR timeout

        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])

    compute_elapsed_time(tic)
