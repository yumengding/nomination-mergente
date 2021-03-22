from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time
import re


url = 'http://sid2nomade.grenet.fr/login?url=http://nouveau.europresse.com/access/ip/default.aspx?un=grenobleT_1'
browser = webdriver.Chrome()
browser.get(url)

#se connecter
time.sleep(3)
browser.find_element_by_xpath('//*[@id="formulaire_nomade"]/form/table/tbody/tr[1]/td[2]/input').send_keys('dingy')
browser.find_element_by_xpath('//*[@id="formulaire_nomade"]/form/table/tbody/tr[2]/td[2]/input').send_keys('207383247qQ')
browser.find_element_by_xpath('//*[@id="formulaire_nomade"]/form/table/tbody/tr[2]/td[3]/input').click()

#le monde

browser.find_element_by_xpath('//*[@id="advLink"]/span').click()
browser.find_element_by_xpath('//*[@id="specific-sources-rd"]').click()
time.sleep(8)
browser.find_element_by_xpath('//*[@id="sourcesFilter"]').send_keys('monde,le')
time.sleep(8)
browser.find_element_by_xpath('//*[@id="filterResult"]/div[13]/label').click()


#date
browser.find_element_by_xpath('//*[@id="DateFilter_DateRange"]/option[11]').click()
browser.find_element_by_xpath('//*[@id="periodRange"]/span[1]/span/select[2]/option[7]').click()
browser.find_element_by_xpath('//*[@id="periodRange"]/span[1]/span/select[1]/option[29]').click()
browser.find_element_by_xpath('//*[@id="periodRange"]/span[1]/span/select[3]/option[181]').click()

browser.find_element_by_xpath('//*[@id="periodRange"]/span[2]/span/select[2]/option[7]').click()
browser.find_element_by_xpath('//*[@id="periodRange"]/span[2]/span/select[1]/option[31]').click()
browser.find_element_by_xpath('//*[@id="periodRange"]/span[2]/span/select[3]/option[181]').click()
browser.find_element_by_xpath('//*[@id="btnSearch"]').click()
time.sleep(7)
browser.find_element_by_xpath('//*[@id="ddlSort"]/option[3]').click()
time.sleep(8)


#charger les pages
temp_height=0
 
while True:
    browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(5)
    check_height = browser.execute_script("return document.documentElement.scrollTop || window.pageYOffset || document.body.scrollTop;")
    if check_height==temp_height:
        break
    temp_height=check_height
    

#article
time.sleep(3)
for a in range(0,1):
	#récupérer la date
	date=browser.find_element_by_xpath('//*[@id="doc'+str(a)+'"]/div[2]/div[2]/div[2]/div[1]/span[1]').text[0:10]
	#accéder le lien
	element = browser.find_element_by_xpath('//*[@id="doc'+str(a)+'"]/div[2]/div[2]/div[1]/a')
	browser.execute_script("arguments[0].click();", element)
	time.sleep(10)

	title=browser.find_element_by_class_name('titreArticleVisu.rdp__articletitle').text
	reg = "[^0-9A-Za-z\s'-éèçàùôûî]"
	title1=re.sub(reg, '', title)

	texte=browser.find_element_by_class_name('docOcurrContainer').text
	if "&" in texte:
		texte1=texte.replace("&","amp;")
	else:
		texte1=texte
	
	author=''
	def isElementExist(browser,element):
		authors=True
		try:
			browser.find_element_by_class_name(element)
			return authors
		except:
			authors=False
			return authors
	
	authors=isElementExist(browser,'docAuthors')
	
	if authors:
		author=browser.find_element_by_class_name('docAuthors').text


	nom=str(a)+"-"+date+"-"+title1
	if len(nom)>150:
		nom=nom[0:150]
	fichier=open(nom+".txt","w",encoding="utf-8")
	fichier.write(texte)
	fichier.close()
	handle=open(nom+".xml","w",encoding="utf-8")
	handle.write('<TEI version="5.0" xmlns="http://www.tei-c.org/ns/1.0">')
	handle.write('\t<teiHeader>')
	handle.write('\t\t<fileDesc>')
	handle.write('\t\t\t<titleStmt>')
	handle.write('\t\t\t\t<title>'+title+'</title>')
	handle.write('\t\t\t\t<author>'+author+'</author>')
	handle.write('\t\t\t</titleStmt>')
	handle.write('\t\t\t<publicationStmt>')
	handle.write('\t\t\t\t<publisher>Le Monde</publisher>')
	handle.write('\t\t\t\t<date>'+date+'</date>')
	handle.write('\t\t\t</publicationStmt>')
	handle.write('\t\t\t<sourceDesc>')
	handle.write('\t\t\t\t<p>La source : Le Monde digital en Europresse.</p>')
	handle.write('\t\t\t</sourceDesc>')
	handle.write('\t\t</fileDesc>')
	handle.write('\t</teiHeader>')
	handle.write('\t<text>')
	handle.write('\t\t<body>')
	handle.write('\t\t\t<p>'+texte1+'</p>')
	handle.write('\t\t</body>')
	handle.write('\t</text>')
	handle.write('</TEI>')
	handle.close()
	browser.back()
	time.sleep(3)

