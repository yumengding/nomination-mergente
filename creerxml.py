def creerxml(title,author,date):
	fichier=open(title+".txt","r")
	contenu=fichier.readlines()
	handle=open(title+".xml","w")

	


	handle.write('<TEI version="5.0" xmlns="http://www.tei-c.org/ns/1.0">')
	handle.write('\t<teiHeader>')
	handle.write('\t\t<fileDesc>')
	handle.write('\t\t\t<titleStmt>')
	handle.write('\t\t\t\t<title>'+title+'</title>')
	handle.write('\t\t\t\t<author>'+author+'</author>')
	handle.write('\t\t\t</titleStmt>')
	handle.write('\t\t\t<publicationStmt>')
	handle.write('\t\t\t\t<date>'+date+'</date>')
	handle.write('\t\t\t</publicationStmt>')
	handle.write('\t\t\t<sourceDesc>')
	handle.write('\t\t\t\t<p>La source : Ressources documentaires - Convention Citoyenne pour le Climat</p>')
	handle.write('\t\t\t</sourceDesc>')
	handle.write('\t\t</fileDesc>')
	handle.write('\t</teiHeader>')
	handle.write('\t<text>')
	handle.write('\t\t<body>')
	for line in contenu:
			if line!="\n":
				line=line.strip()
				if line[0]==" ":
					line=line[1:]
				if len(line)>3 and line.isdigit()==False:
					handle.write("<p>"+line+"</p>"+"\n\n")
	handle.write('\t\t</body>')
	handle.write('\t</text>')
	handle.write('</TEI>')
	handle.close()
	fichier.close()
	
title="Synthèse des recommandations du Haut Conseil pour le Climat à destination de la Convention"
author="ami"
date="0909"
creerxml(title,author,date)
