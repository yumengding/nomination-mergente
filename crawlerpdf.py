import importlib
import sys
import re

 
importlib.reload(sys)

 
import os.path
from pdfminer.pdfparser import  PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
 
text_path = r"Stratégie Nationale Bas-Carbone_SNBC-2-en-4-pages_-web.pdf"
 
def parse():

    fp = open(text_path,'rb')
    parser = PDFParser(fp)
    doc = PDFDocument()
    parser.set_document(doc)
    doc.set_parser(parser)
    doc.initialize()
 
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr,laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr,device)

        for page in doc.get_pages():
            interpreter.process_page(page)
            layout = device.get_result()
            for x in layout:
                if(isinstance(x,LTTextBoxHorizontal)):
                    with open(r"Synthèse des recommandations du Haut Conseil pour le Climat à destination de la Convention.txt",'a') as f:
                        results = x.get_text()  
                        text=re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+",u"",results)                      
                        f.write(text  +"\n")
 
if __name__ == '__main__':
    parse()
   
