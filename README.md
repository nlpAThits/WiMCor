# Wikipedia Metonymy Corpus

Code for the paper "A Large Harvested Corpus of Location Metonymy" published in LREC 2020.

## Data

[WiMCor](https://kevinalexmathews.github.io/software/)

## Running the code (WIP!)

1. Generate metonymic pairs

```$ python pathfinder.py -disamb_file ./disambiguation_page_titles -vehicles 'PopulatedPlace' -targets 'Q3918'```

where `disamb_file` is a file consisting of titles of Wikipedia disambiguation pages, one per line. 

## License

[GNU GPLv3](LICENSE)
