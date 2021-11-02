import os
import json

DATAROOT = "./"
TRAINJSON = os.path.join(DATAROOT,"train.json")
VALIDJSON = os.path.join(DATAROOT,"val.json")
TESTJSON = os.path.join(DATAROOT,"test.json")

'''
Edit file name as img id
'''
def _edit_file_name(json_dir, annotationPath):

	with open(json_dir, "r", encoding="utf8") as outfile:
		datas = json.load(outfile)

		for data in datas["images"]:
			data["file_name"] = f"{data['id']:04}.jpg"

	with open(annotationPath+"/edit.json", "w") as new_file:
		json.dump(datas, new_file, indent='\t')

'''
Wrap func
'''
def make(json, path):
	annotationPath = '../mmseg/annotations/'+path

	os.makedirs(annotationPath, exist_ok=True)
	_edit_file_name(json, annotationPath)


'''
Main
'''
def __main__():

	make(TRAINJSON, 'training')
	make(VALIDJSON, 'validation')
	make(TESTJSON, 'test')

	
if __name__=='__main__':
	__main__()
