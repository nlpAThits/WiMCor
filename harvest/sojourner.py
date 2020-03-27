import re
import sys
import wikipediaapi
from SPARQLWrapper import SPARQLWrapper, JSON
from bs4 import BeautifulSoup
from itertools import combinations, product
from multiprocessing import Pool as Threadpool
from multiprocessing import cpu_count

from services import get_tic, compute_elapsed_time

en_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
'''
DBpedia endpoints:
   http://dbpedia-live.openlinksw.com/sparql
   http://live.dbpedia.org/sparql
   http://dbpedia.org/sparql
   http://dbpedia.org/snorql
'''
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

def get_remote_links(title):
    return en_wiki.page(title).links.keys()

def get_remote_summary(title):
    return en_wiki.page(title).summary

def get_dbpedia_categories(mention):
    """
        To get all categories under which a page belongs
    """
    mention = mention.replace(' ', '_')

    query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE { <http://dbpedia.org/resource/""" + mention + """> rdf:type ?label
            }"""
    sparql.setQuery(query)

    try:
        results = sparql.query().convert()["results"]["bindings"]
    except Exception:
        results = []

    categories = [category["label"]["value"].split('/')[-1] for category in results]
    return categories

def matches_association_element(page, association_elements):
    categories = get_dbpedia_categories(page)
    return any(x in categories for x in association_elements)

def matches_association_pair(vehicle, target, associations):
    categories_vehicle = get_dbpedia_categories(vehicle)
    categories_target = get_dbpedia_categories(target)

    is_valid = False
    '''
        to identify valid metonymic pair:
        Hit Abu Ghraib (disambiguation): (<Abu Ghraib>, <Abu Ghraib prison>)
    '''
    for (cat_vehicle, cat_target) in product(categories_vehicle, categories_target):
        if (cat_vehicle, cat_target) in associations:
            is_valid = True

    '''
        to filter out noisy pairs such as:
        Hit Abbottabad (disambiguation): (<Abbottabad District>, <Abbottabad>
    '''
    for cat_target in categories_target:
        if cat_target in [association[0] for association in associations]:
            is_valid = False

    return is_valid

