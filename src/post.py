import requests
import json
#r = requests.post("http://9.24.175.45:8081/api/test", json={"data": "但是我们也不得不承认，当你把电影从梦变成现>    实，再挑剔的观众也不会吝啬自己的赞美之词。", "keywords": ["挑剔"]})
#r = requests.post("http://11.214.30.194:6003/ver0", json = {"instruction": "给我写一个新年快乐的祝福", "input":"","output":""})

fin = open("/apdcephfs/share_916081/victoriabi/para_data/test_data/test_key.txt", "r")

for line in fin.readlines():

    #keys = "人大代表,团结"
    keys = line.strip()
    r = requests.post("http://11.214.30.194:6003/ver0", json = {"instruction":keys, "input":"","output":""})
    data = json.loads(r.text)
#data=r.text
    print("inputs:", keys)
    print("instruction only:")
    print(data["BLOOMZ+para-data-5w_v2_response"])
    r = requests.post("http://11.214.30.194:6004/ver1", json = {"instruction":keys, "input":"","output":""})
    data = json.loads(r.text)
#data=r.text
    print("finetuning results:")
    print(data["BLOOMZ+para-data-5w_v2_response"])

