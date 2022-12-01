import urllib2
import time
import sys
import os
import io
import pickle

inputfile = ''
PubTator_username = ''
url_Submit = ''
email = ''

#Submit

url_Submit = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/tmChem/Submit/"

def tag(inputfile):
	fh = open(inputfile)
	InputSTR=''
	for line in fh:
		InputSTR = InputSTR + line

	urllib_submit = urllib2.urlopen(url_Submit, InputSTR)
	urllib_result = urllib2.urlopen(url_Submit, InputSTR)
	SessionNumber = urllib_submit.read()

	
	print("Thanks for your submission. The session number is : "+ SessionNumber + "\n")
	print("\nThe request is received and processing....\n\n")
	#Receive
	url_Receive = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/" + SessionNumber + "/Receive/"
	
	code=404
	while(code == 404 or code == 501):
		time.sleep(5)
		try:
			urllib_result = urllib2.urlopen(url_Receive)
		except urllib2.HTTPError as e:
			code = e.code
		except urllib2.URLError as e:
			code = e.code
		else:
			code = urllib_result.getcode()

	return urllib_result.read()

def process_result(text):
	chem = set()
	lines = text.split('\n')
	for l in lines:
		if(l.strip()!="" and l[8:11] not in ('|t|', '|a|')):
			k = l.split("\t")
			if(k[4]=="Chemical" and len(k[3])>3):
				chem.add(k[3])
	return chem



path = "/home/anjadhav/pubtator/dev"
store_path = "/home/anjadhav/pubtator/tag_result/dev"


files =os.listdir(path)
done = os.listdir(store_path)
# files = ['/home/anjadhav/Chemical-Patent-Reaction-Extraction/RESTfulAPI.client/examples/ex.PubTator']
files = [fd for fd in files if fd.split('.')[0]+".pkl" not in done]
print(files)
for fp in files:
	print(fp)
	try:
		result = tag(os.path.join(path, fp))
		result = process_result(result)
		
		with open(os.path.join(store_path, fp.split('.')[0]+".pkl"), 'wb') as f:
			pickle.dump(list(result), f)
		# wp = io.open(os.path.join(store_path, f), mode='w', encoding= 'utf-8' )
		# wp.write(result.decode('utf-8'))
		# wp.close()
		# with open('parrot.pkl', 'rb') as f:
		# 	mynewlist = pickle.load(f)
		# 	print(mynewlist)

		print("Written file: ", fp)
	except Exception as err:
		print("Error occucured")
		print(err)