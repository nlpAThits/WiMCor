#!/usr/bin/python3
import sys
import requests
from bs4 import BeautifulSoup
from bs4 import Tag
from bs4 import NavigableString
from multiprocessing import Pool as Threadpool
from multiprocessing import cpu_count
from multiprocessing import Lock
import random
import spacy
from services import get_tic, compute_elapsed_time

nlp = spacy.load('en_core_web_sm')
s = requests.Session()
url = 'https://en.wikipedia.org/w/api.php'
directory = sys.argv[1]
num_backlinks = sys.argv[2]

def get_text(title):
    '''
        To get the text of a Wikipedia page.
        https://www.mediawiki.org/wiki/API:Parsing_wikitext
    '''
    text = ''

    try:
        params = {'action': 'parse', 'format': 'json', 'page': title}
        r = s.get(url=url, params=params)
        data = r.json()

        text = data['parse']['text']['*']
    except:
        print('Exception caught: Text of article {} not extracted!'.format(title))

    return text

def get_backlinks(title):
    '''
        To get the backlinks of a Wikipedia page.
        https://www.mediawiki.org/wiki/API:Backlinks

        'bdirect'
            In the script below,
                only direct links are returned.
            When blredirect is set,
                the response will include any pages which backlink to redirects for the value in bltitle.

        'bllimit'
            The parameter bllimit specifies how many total pages to return.
            No more than 500 (5,000 for bots) allowed.
            Default: 10

        'blnamespace'
            https://en.wikipedia.org/wiki/Wikipedia:Namespace
            0 for (Main/Article)
            1 for Talk on Main/Article
            2 for User
            3 for Talk on User
            10 for Template
            14 for Category
            ...

    '''

    params = {'action': 'query', 'format': 'json', 'list': 'backlinks', 'blnamespace': 0, 'bllimit':num_backlinks, 'bltitle': title}
    r = s.get(url=url, params=params)
    data = r.json()

    backlinks = data['query']['backlinks']
    return backlinks

def generate_sample(text, target_page_title, anchor_text, label, cat):
    '''
        Generate sample from a valid paragraph
    '''

    link_to_target_page = text.find('a', {'title' : target_page_title})

    left_context = ''
    for foo in link_to_target_page.previous_siblings:
        if isinstance(foo, Tag) and foo.name != 'sup':
            left_context = foo.text + left_context
        elif isinstance(foo, NavigableString):
            left_context = foo + left_context

    pmw = link_to_target_page.text

    right_context = ''
    for foo in link_to_target_page.next_siblings:
        if isinstance(foo, Tag) and foo.name != 'sup':
            right_context = right_context + foo.text
        elif isinstance(foo, NavigableString):
            right_context = right_context + foo

    sentence = left_context + pmw + right_context

    doc = nlp(sentence)
    start_pmw = len(nlp(left_context))
    end_pmw = start_pmw + len(nlp(pmw))

    '''
        Filter out noisy paragraphs.
        1. Paragraphs that are either too short or too long.
        2. Paragraphs having the anchor text in the left or right context.
        3. Paragraphs whose backlink text do not exactly match the title of the target page.
        4. Paragraphs in which the backlink text is part of a noun compound.
    '''
    has_invalid_length = (len(doc) < 10 or len(doc) > 512)
    has_anchor_text = (anchor_text in (left_context + right_context))
    not_exact_match = (pmw != target_page_title)
    is_in_compound = any( [token.dep_ == 'compound' and token.head.i not in range(start_pmw, end_pmw) for token in doc[start_pmw: end_pmw]]
                        + [token.dep_ == 'compound' and token.head.i in range(start_pmw, end_pmw) for token in doc[:start_pmw - 1]]
                        + [token.dep_ == 'compound' and token.head.i in range(start_pmw, end_pmw) for token in doc[end_pmw:]]
                        )

    if has_invalid_length or has_anchor_text or not_exact_match or is_in_compound:
        return ''

    '''
        If a valid paragraph is identified, perform required changes to generate the sample.
    '''
    has_article = any([token.dep_ == 'det' and token.head.i in range(start_pmw, end_pmw) for token in doc])
    label = 'met' if label == 1 else 'lit'

    if has_article:
        det_index = [token.i for token in doc if token.dep_ == 'det' and token.head.i in range(start_pmw, end_pmw)][0]
        lc = nlp(left_context)[:det_index]
        if nlp(left_context)[det_index+1:]:
            lc = '{} {}'.format(lc, nlp(left_context)[det_index+1:])
        sample = '<sample>{} <pmw target=\'{}\' label=\'{}\' category=\'{}\'>{}</pmw>{}</sample>'.format(lc, target_page_title, label, cat, anchor_text, right_context)
    else:
        sample = '<sample>{} <pmw target=\'{}\' label=\'{}\' category=\'{}\'>{}</pmw>{}</sample>'.format(left_context, target_page_title, label, cat, anchor_text, right_context)

    return sample

def check_paragraph(text, target_page_title, anchor_text, label, cat):
    '''
        To check whether a paragraph meets the requirements to be  sample.

        1. paragraph has a backlink                         e.g. Heidelberg
        2. matching instance does NOT have an article       e.g. the Heidelberg
        3. matching instance is NOT part of a compound noun e.g. Heidelberg Wolves
    '''

    target_page_url = '/wiki/' + target_page_title.replace(' ', '_')
    links = [link['href'] for link in text.find_all('a') if 'href' in link.attrs.keys()]

    if target_page_url not in links:
        return ''

    sample = generate_sample(text, target_page_title, anchor_text, label, cat)
    return sample

def soupify(text, page, anchor_text, label, cat):
    '''
        To soupify the text of a Wikipedia
            and identify the paragraphs containing the backlinks.
    '''

    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = soup.find_all('p')

    sample = ''
    for paragraph in paragraphs:
        sample = check_paragraph(paragraph, page, anchor_text, label, cat)
        if  sample != '':
            '''
                Generally only the first instance has backlink.
                Hence the first paragraph matching rest of the condition is a sample.
                So break the loop to avoid checking rest of the paragraphs.
            '''
            break

    return sample

def logger(page, samples):
    lock = Lock()
    lock.acquire()
    for sample in samples:
        print('{}'.format(sample.replace('\n', ' ').strip()))
    lock.release()

    return

def get_labelwise_samples(page, anchor_text, label, cat):
    backlinks = get_backlinks(page)

    samples = []
    for backlink in backlinks:
        text = get_text(backlink['title'])
        sample = soupify(text, page, anchor_text, label, cat)
        if sample != '':
            samples.append(sample)
    logger(page, samples)

    return

def extractor(line):
    anchor_text = line[0].strip().replace(' (disambiguation)', '')
    neg = line[1].strip()
    pos = line[2].strip()
    # anchor_text = 'Albion'
    # neg = 'Kurukshetra'
    # pos = 'Kurukshetra War'

    get_labelwise_samples(pos, anchor_text, label=1, cat='FACILITY')
    get_labelwise_samples(neg, anchor_text, label=0, cat='LOCATION')

    return

if __name__=="__main__":
    tic = get_tic()

    with open('{}anchors'.format(directory)) as fp:
        anchors = fp.readlines()

    # Negatives
    with open('{}vehicles'.format(directory)) as fp:
        vehicles = fp.readlines()

    # Positives
    with open('{}targets'.format(directory)) as fp:
        targets = fp.readlines()

    '''
    lines = list(zip(anchors, vehicles, targets))
    extractor(random.choice(lines))
    '''

    # Create a threadpool for reading in files
    threadpool = Threadpool(20)

    # Read in the files as a list of lists
    results = threadpool.map(extractor, zip(anchors, vehicles, targets))

    threadpool.close()
    threadpool.join()

    compute_elapsed_time(tic)
