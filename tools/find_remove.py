from xml.dom.minidom import parse
import xml.dom.minidom
import os.path as osp
# 使用minidom解析器打开 XML 文档
root = './data/ulcer_fine/before'
name = 'filter11'
DOMTree = xml.dom.minidom.parse(osp.join(root, name + '.xml'))
collection = DOMTree.documentElement 
# 在集合中获取所有电影
images = collection.getElementsByTagName("image")
txt_file = osp.join(root, name + '.txt')

des = {}
with open(txt_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        print(line)
        des[int(line[0])] = line[1]

remove1_txt_f = open(osp.join(root, name + 'shidao.txt'), 'w')
remove2_txt_f = open(osp.join(root, name + 'no.txt'), 'w')
remain_all_txt_f = open(osp.join(root, name + 'fine.txt'), 'w')
remove_txt_f = open(osp.join(root, name + 'remove.txt'), 'w')

# 打印每部电影的详细信息
for image in images:
    id = int(image.getAttribute("id"))
    name = image.getAttribute("name")
    name = name.split('/')[-1]
    if id in des and ('未标注' in des[id] or '删除'  in des[id]):
        remove2_txt_f.write(name + '\n')
        remove_txt_f.write(name + '\n')
    elif id in des and '食管' in des[id]:
        remove1_txt_f.write(name + '\n')
        remove_txt_f.write(name + '\n')
    else:
        remain_all_txt_f.write(name + '\n')

remove1_txt_f.close()
remove2_txt_f.close()
remove_txt_f.close()
remain_all_txt_f.close()