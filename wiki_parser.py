import cPickle as pkl 
import re
import parameters as prm

class Parser():

    def __init__(self):

        self.f = open(prm.dump_path, "rb")
        if prm.compute_page_pos:
            self.page_pos, self.cat_pages = self.precompute_idx()
            with open(prm.page_pos_path, "wb") as f:
                pkl.dump(self.page_pos, f)
            with open(prm.cat_pages_path, "wb") as f:
                pkl.dump(self.cat_pages, f)

        else:
            with open(prm.page_pos_path, "rb") as f:
                self.page_pos = pkl.load(f)
            with open(prm.cat_pages_path, "rb") as f:
                self.cat_pages = pkl.load(f)


    def get_links(self, text):
        '''
        Return a dictionary of links found in the input text. A link is always between '[[' and ']]' .
        '''
        links = [] 
        start = -1
        for i in range(len(text)-2):
            if text[i:i+2] == "[[":
                start = i+2
                
            if start >= 0 and text[i:i+2] == "]]":
                link = text[start:i]
                link = link.split("|")[0].strip() # Get only the first part of the link.
               
                if link in self.page_pos:  # Only add link if it is in the list of pages.
                    links.append(link.lower())

                start = -1
        return links


    def remove_recursively(self, text, st_tag, end_tag, st_op=None):
        '''
        Remove recursively all the text that are in st_tag and end_tag, including the tags.
        '''
        tag_count = 0
        out = ''
        cc = len(text)
        if not st_op:
            st_op = st_tag
            
        a = [m.start() for m in re.finditer(st_tag, text)] + [len(text)]
        b = [m.start() for m in re.finditer(st_op, text)] 
        b = list(set(b) - set(a)) + [len(text)]
        c = [m.start() for m in re.finditer(end_tag, text)] + [len(text)]
        a.sort()
        b.sort()
        c.sort()
        st_tag = st_tag.replace('\\','')
        end_tag = end_tag.replace('\\','')
        st_op = st_op.replace('\\','')
        
        i = 0
        ni = 0
        while i < cc:
            if b[0] < a[0]:
                if tag_count > 0:
                    r = b
                else:
                    del b[0]
                    r = a
            else:
                r = a 
            if r[0] < c[0]:
                ni = r[0]
                if tag_count == 0:
                    out += text[i:ni]
                tag_count += 1
                del r[0]
            elif c[0] < r[0]:
                if tag_count > 0:
                    tag_count -= 1
                    if tag_count == 0:
                        i = c[0]+len(end_tag)
                del c[0]
                
            elif c[0] == len(text) and r[0] == len(text):
                out += text[i:len(text)]
                break
                
        return out


    def precompute_idx(self):

        page_pos = {}
        cat_pages = {}
        textbegin = False
        title = ''
        text = ''
        n = 0
        f = open(prm.dump_path, 'rb')

        while True:
            #don't use for line in f because tell() does not return the right position
            line = f.readline()
            if (line == ''):
                break

            if ("<page>" in line):
                pagebegin = True
                textbegin = False
                title = ""
                pos = f.tell()

            if ("</page>" in line):
                pagebegin = False
                textbegin = False
                title = ""

            if ("<title>" in line) and ("</title>" in line) and pagebegin:
                title = line.replace("    <title>","").replace("</title>\n","").lower()

                if n % 10000 == 0:
                    print n
                n += 1

                #if n > 1000000:
                #    break

            if textbegin:
                line_clean = line.replace("</text>", "")
                text.append(line_clean)
    
            if ("<text xml:space=\"preserve\">" in line) and pagebegin:
                textbegin = True
                line_clean = line.replace("      <text xml:space=\"preserve\">","")
                text = [line_clean]

            if ('</text>' in line) and pagebegin:
                pagebegin = False
                textbegin = False

                # don't add if this is a redirect page.
                skip = False
                for p in text:
                    if p.strip().lower().startswith('#redirect'):
                        skip = True
                        break

                if skip:
                    continue

                #do not add pages that start with 'Wikipedia:', 'file:', 'image:' or 'template'.
                if title.startswith('wikipedia:'):
                    continue
                if title.startswith('file:'):
                    continue
                if title.startswith('image:'):
                    continue
                if title.startswith('template:'):
                    continue

                if title not in page_pos:
                    page_pos[title] = pos

                for p in text:
                    if p.startswith('[[Category:'): # get the categories
                        catname = p.strip().split('|')[0].replace('[[', '').replace(']]', '').lower()
                        if catname not in cat_pages:
                            cat_pages[catname] = {}
                        cat_pages[catname][title] = 0 
        f.close()
        return page_pos, cat_pages


    def parse(self, title):

        pos = self.page_pos[title]
        self.f.seek(pos)
        pagebegin = True
        textbegin = False
        sections = []

        while True:
            line = self.f.readline()
            if (line == ''):
                break
            line = line
        
            if ("</page>" in line):
                pagebegin = False
                break

            if line.strip()[:2] == "==" and line.strip()[-2:] == "==" and line.strip()[:3] != "===" and line.strip()[-3:] != "===": # another section begins...
                sections.append({"text": line})
                continue #do not add the section twice.

            if textbegin:
                if not line.startswith("[[Category:"): # skip the categories
                    line_clean = line.replace("</text>", "")
                    sections[-1]["text"] += line_clean

            if ("<text xml:space=\"preserve\">" in line) and pagebegin:
                textbegin = True
                line_clean = line.replace("      <text xml:space=\"preserve\">","").replace("</text>", "")
                sections.append({"text": line_clean}) #add a section, it will be the abstract

            if ("</text>" in line) and pagebegin:
                #st1=time.time()
                textbegin = False

                if 'category:' in title: # if the page is a category page, add section with links to categories
                    if title in self.cat_pages:
                        # add a section that contains the link to subcategories and pages.
                        lp = []
                        for page in self.cat_pages[title].keys():
                            lp.append('[[' + page + ']]\n')
                        lp.sort()
                        sections.append({'text': ''.join(lp)})

                text = ''
                for section in sections:
                    # do not add some specific sections:
                    skip = False
                    for exsec in ['References', 'External links', 'Bibliography', 'Partial bibliography', 'See also', 'Further reading', 'Notes', 'Additional sources']:
                        if section["text"].replace('===','').replace('==','').strip().startswith(exsec):
                            skip = True
                    if skip:
                        continue

                    text += ' ' + section["text"]

                # clean text
                #st2 = time.time()

                text = self.remove_recursively(text, '\{\{', '\}\}')
                text = self.remove_recursively(text, '\{\|', '\|\}')
                text = self.remove_recursively(text, '\&lt\;\!\-\-', '\-\-\&gt\;')
                text = self.remove_recursively(text, '\<gallery\>', '\<\/gallery\>')
                text = self.remove_recursively(text, '\[\[File\:', '\]\]', st_op='\[\[')
                text = self.remove_recursively(text, '\[\[file\:', '\]\]', st_op='\[\[')
                text = self.remove_recursively(text, '\[\[Image\:', '\]\]', st_op='\[\[')
                text = self.remove_recursively(text, '\[http\:', '\]', st_op='\[')
                text = self.remove_recursively(text, '\[https\:', ']', st_op='\[')
                text = re.sub(r'\&lt\;ref.*?\&lt\;\/ref\&gt\;', '', text, flags=re.DOTALL)
                text = re.sub(r'\&lt\;ref.*?\/\&gt\;', '', text, flags=re.DOTALL)
                #st3 = time.time()
                # Get links
                links = self.get_links(text)

                return text, links 

        return None

