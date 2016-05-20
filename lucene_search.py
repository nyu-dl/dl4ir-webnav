'''
Use Lucene to retrieve candidate documents for given a query.
'''

import sys
import shutil
import os
import lucene
import wiki
import parameters as prm
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser


def create_index():

    lucene.initVM()
    if os.path.exists(prm.index_folder):
        shutil.rmtree(prm.index_folder)

    indexDir = SimpleFSDirectory(File(prm.index_folder))
    writerConfig = IndexWriterConfig(Version.LUCENE_4_10_1, StandardAnalyzer())
    writer = IndexWriter(indexDir, writerConfig)
    wk = wiki.Wiki(prm.pages_path)

    print "%d docs in index" % writer.numDocs()
    print "Reading files from wikipedia..."
    n = 0
    for l in wk.get_text_iter():
        doc = Document()
        doc.add(Field("text", l, Field.Store.YES, Field.Index.ANALYZED))
        doc.add(Field("id", str(n), Field.Store.YES, Field.Index.ANALYZED))
        writer.addDocument(doc)
        n += 1
        if n % 100000 == 0:
            print 'indexing article', n
    print "Indexed %d docs from wikipedia (%d docs in index)" % (n, writer.numDocs())
    print "Closing index of %d docs..." % writer.numDocs()
    writer.close()



def get_candidates(qatp):

    if prm.create_index:
        create_index()

    lucene.initVM()
    analyzer = StandardAnalyzer(Version.LUCENE_4_10_1)
    reader = IndexReader.open(SimpleFSDirectory(File(prm.index_folder)))
    searcher = IndexSearcher(reader)
    candidates = []
    n = 0
    for q,a,t,p in qatp:
        if n % 100 == 0:
            print 'finding candidates sample', n
        n+=1

        q = q.replace('AND','\\AND').replace('OR','\\OR').replace('NOT','\\NOT')
        query = QueryParser(Version.LUCENE_4_10_1, "text", analyzer).parse(QueryParser.escape(q))
        hits = searcher.search(query, prm.max_candidates)
        c = []
        for hit in hits.scoreDocs:
            doc = searcher.doc(hit.doc)
            c.append(doc.get("id"))

        candidates.append(c)
        
    return candidates

