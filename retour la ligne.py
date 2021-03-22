fichier=open("Synthèse des recommandations du Haut Conseil pour le Climat à destination de la Convention.txt","r")
contenu=fichier.readlines()
a=''
for line in contenu:

	if line=="\n":
		line="*"
		a=a+line.strip()
	else:
		a=a+line.strip()

b=a.replace("*","\n\n")
fichier.close()
handle1=open("Synthèse des recommandations du Haut Conseil pour le Climat à destination de la Convention2.txt","w")
handle1.write(b)
handle1.close()
