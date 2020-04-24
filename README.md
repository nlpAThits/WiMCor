# Wikipedia Metonymy Corpus

Code for the paper "A Large Harvested Corpus of Location Metonymy" published in LREC 2020.

## Data

[WiMCor](https://kevinalexmathews.github.io/software/)

## Run the code (WIP!)

1. Generate metonymic pairs

```$ python pathfinder.py -disamb_file ./disambiguation_page_titles -vehicles 'PopulatedPlace' -targets 'Q3918'```

where `disamb_file` is a file consisting of titles, one per line, of Wikipedia disambiguation pages.
This command extracts metonymic pairs of the form `<vehicle>-for-<target>` from the offline version (XML dumps) and the online version (MediaWiki).

2. Generate samples

```$ python gen_samples.py -directory ./```

where `directory` denotes the directory having the output of log processed by `process-log.sh`.
This command generates the annotated samples in XML format.

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
