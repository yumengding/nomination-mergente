import string
def trouverinconnu(title):
	fichier=open(title+"2.txt","r")
	fichier=fichier.readlines()
	
	handle=open(title+".txt","w")
	
	contenu=open("ABU.csv","r")
	listeflechie=[]
	for line in contenu :	
		if "\t" in line:			
			line=line.strip()		
			coupe = line.split("\t")
			flechie = coupe[0]	
			listeflechie.append(flechie)

	
	for ligne in fichier:
		phrase=''
		if ligne!="\n":
			ligne=ligne.strip()
			if " " in ligne:
				coupe=ligne.split(" ")
				for elt in coupe:
					elt=elt.strip()
					if elt.isdigit()==False and elt not in string.punctuation:
						elt1=elt.lower()
						ponc1=''
						ponc2=''
						if elt1[0] in string.punctuation:
							ponc1=elt1[0]
							if elt1[-1] in string.punctuation:
								ponc2=elt1[-1]
						elif elt1[-1] in string.punctuation:
							ponc2=elt1[-1]
						elt2 = elt1.strip(string.punctuation)
						if elt2 not in listeflechie :
							i=0
							find=False
							while i<len(elt2) and not find:
								if elt2[:i] in listeflechie and elt2[i:] in listeflechie:
									mot1=elt2[:i]
									mot2=elt2[i:]
									find=True
									newmot=ponc1+mot1+" "+mot2+ponc2
									phrase=phrase+newmot+" "
								else:
									i=i+1
							if find==False:
								unknown=ponc1+elt2+ponc2
								phrase=phrase+unknown+" "
						else:
							phrase=phrase+elt+" "
					else:
						phrase=phrase+elt+" "
			else:			
				phrase=phrase+elt+" "
		handle.write(phrase+"\n")			

title=""
trouverinconnu(title)
