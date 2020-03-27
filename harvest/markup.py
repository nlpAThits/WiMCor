#!/usr/bin/env python3

import re, sys, html


def repl(a, b, s):
    while True:
        s2 = re.sub(a, b, s)
        if s2 == s:
            return s
        s = s2

BLUE = "\033[1;34m"
BOLD = "\033[;1m"
ITALIC = "\033[;3m"
RESET = "\033[0;0m"


def change(s):
    # Links.
    s = repl("\[\[([^\[\]]*)\|([^\[\]]*)\]\]", BLUE + "\\2" + RESET, s)
    s = repl("\[\[([^\[\]]*)\]\]", BLUE + "\\1" + RESET, s)
    # Bold / italic
    s = repl("'''([^']*)'''", BOLD + "\\1" + RESET, s)
    s = repl("''([^']*)''", ITALIC + "\\1" + RESET, s)
    # Annoying references - yeah, I know XML is not parsable by regex, but this
    # is better than nothing ^_^'
    s = repl("<ref([^<]*)</ref>", "", s)
    s = repl("<ref([^>]*)/>", "", s)

    s = html.unescape(s)
    return s


if __name__ == "__main__":
    print(change(sys.stdin.read()))
