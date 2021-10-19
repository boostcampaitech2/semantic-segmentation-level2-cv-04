import os
import json

DATAROOT = "./"
TRAINJSON = os.path.join(DATAROOT,"train.json")
VALIDJSON = os.path.join(DATAROOT,"val.json")
TESTJSON = os.path.join(DATAROOT,"test.json")

'''
Edit file name as img id
'''
def _edit_file_name(json_dir, imagePath):
    with open(json_dir, "r", encoding="utf8") as outfile:
        data = json.load(outfile)
        data["file_name"] = f"{data['id']:04}.jpg"

    with open(imagePath, "w") as new_file:
        json.dump(data, new_file, indent='\t')

'''
Make dir
'''
def _make_directory(paths):
	for path in paths:
		os.makedirs(path, exist_ok=True)


'''
Wrap func
'''
def make(json,path):
	imagePath = '../mmseg/annotations/'+path

	_make_directory(imagePath)
	_edit_file_name(json,imagePath)


'''
Main
'''
def __main__():

	make(TRAINJSON, 'training')
	make(VALIDJSON, 'validation')
	make(TESTJSON, 'test')