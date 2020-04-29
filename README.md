# Wikipedia Metonymy Corpus

Code for the paper "A Large Harvested Corpus of Location Metonymy" published in LREC 2020.

## Data

[WiMCor](https://kevinalexmathews.github.io/software/)

## Run the code

1. Generate metonymic pairs

```$ python -u gen_metpairs.py -disamb_file ./disambiguation_page_titles -vehicles 'PopulatedPlace' -targets 'Q3918'```

where `disamb_file` is a file consisting of titles, one per line, of Wikipedia disambiguation pages.
This command extracts metonymic pairs of the form `<vehicle>-for-<target>` from the offline version (XML dumps) and the online version (MediaWiki). 
Check out 
[here](https://wiki.dbpedia.org/services-resources/datasets/dbpedia-datasets#h434-6),
[here](http://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=SELECT+DISTINCT+%3Fcategory%0D%0AWHERE+%7B%3Farticle+rdf%3Atype+%3Fcategory+.%7D%0D%0ALIMIT+1000000&format=text%2Fhtml&CXML_redir_for_subjs=121&CXML_redir_for_hrefs=&timeout=30000&debug=on&run=+Run+Query+)
and 
[here](https://www.wikidata.org/wiki/Wikidata:Item_classification) for different types of categories that can be used as vehicles and targets.

2. Generate samples

```$ python gen_samples.py -directory ./```

where `directory` denotes the directory having the output of list of metonymic pairs processed by `process-pairs.sh`.
This command generates the annotated samples in XML format.

3. Run baselines (WIP!)

The scripts are available in the directories `glove/` and `bert/`.
These script are based on [this](https://github.com/milangritta/Minimalist-Location-Metonymy-Resolution) repository.

First create pickle files for each annotated file with the command:

```$ python get_pickle.py -c imm -f filepath```

Then train and test the LSTM model using the command:

```$ python get_results.py -c imm -w 5 -d directorypath```

where `directorypath` denotes the path to the directory containing the pickle files. Repeat the same for prewin as well. We have provided a few annotated files alongside to play with.

## Cite the paper

```
@inproceedings{lrec20-wimcor,
author    = {Mathews, Kevin Alex and Strube, Michael},
booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2020)},
publisher = {European Languages Resources Association (ELRA)},
title     = {A Large Harvested Corpus of Location Metonymy},
year      = {2020}
}
```

## License

[GNU GPLv3](LICENSE)
