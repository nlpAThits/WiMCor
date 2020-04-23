#!/usr/bin/env python3
import zipfile
import bz2
import sys
import html
from xml.etree.ElementTree import XMLParser
import markup


def build(orig_index, new_index):
    bz = bz2.BZ2File(orig_index, "r")

    print("Decompressing and parsing index...")
    lines = []
    s = b""
    while True:
        rd = bz.read(65536)
        if not rd:
            break
        s += rd
        arr = s.split(b"\n")
        for line in arr[:-1]:
            f = line.find(b":")
            f2 = line.find(b":", f + 1) + 1
            lines.append(line[f2:] + b"\x00" + line[:f])
            if len(lines) % 100000 == 0:
                print("%d pages..." % len(lines))
        s = arr[-1]

    print("Sorting index lines...")
    lines.sort()

    print("Creating index...")
    PER_FILE = 2000
    zf = zipfile.ZipFile(new_index, "w", zipfile.ZIP_DEFLATED)
    index = 0
    i = 0
    while index < len(lines):
        if index % 100000 == 0:
            print("%d/%d..." % (index, len(lines)))
        zf.writestr("%08d" % i, b"\n".join(lines[index:index+PER_FILE]))
        index += PER_FILE
        i += 1

    print("Finalizing index...")
    zf.close()


def read(indexzf, which):
    f = indexzf.open(which)
    res = {}
    for line in f.readlines():
        line = line.decode("utf-8").strip()
        colon = line.rfind("\x00")
        offset = int(line[colon+1:])
        name = html.unescape(line[:colon])
        res[name] = offset
    return res


def get(datafile, indexfile, query=None, raw=False):
    zf = zipfile.ZipFile(indexfile, "r")
    files = sorted(zf.namelist())
    if query is None:
        for fname in files:
            for name in sorted(read(zf, fname).keys()):
                print(name)
        return

    lo = 0
    hi = len(files) - 1
    found = None
    while lo < hi:
        mi = (lo + hi) // 2
        name2offset = read(zf, files[mi])
        if query in name2offset:
            found = name2offset[query]
            break
        else:
            key = next(iter(name2offset))
            if query < key:
                hi = mi - 1
            else:
                lo = mi + 1

    if found is None:
        name2offset = read(zf, files[lo])
        if query in name2offset:
            found = name2offset[query]

    zf.close()

    if found is None:
        print("Not found...")
        return

    dec = bz2.BZ2Decompressor()
    df = open(datafile, "rb")
    df.seek(found)


    class Parser(object):
        def __init__(self):
            self.stack = []
            self.done = False

        def start(self, tag, attrib):
            if self.done: return
            self.stack.append(tag)
            if tag == "title":
                self.title = ""
            elif tag == "text":
                self.text = ""

        def end(self, tag):
            if self.done: return
            self.stack.pop()
            if tag == "text" and self.title == query:
                self.done = True

        def data(self, data):
            if self.done: return
            if self.stack[-1] == "title":
                self.title += data
            elif self.stack[-1] == "text":
                self.text += data


    target = Parser()
    parser = XMLParser(target=target)
    parser.feed("<root>")
    while not target.done:
        data = dec.decompress(df.read(65536))
        parser.feed(data)

    text = target.text
    if not raw:
        text = markup.change(text)
    return target.title, text


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        orig_index = sys.argv[2]
        new_index = sys.argv[3]
        build(orig_index, new_index)
    elif cmd == "get":
        datafile = sys.argv[2]
        indexfile = sys.argv[3]
        query = sys.argv[4]
        raw = "--raw" in sys.argv
        get(datafile, indexfile, query, raw)
    elif cmd == "list":
        datafile = sys.argv[2]
        indexfile = sys.argv[3]
        get(datafile, indexfile)
    else:
        usage()

