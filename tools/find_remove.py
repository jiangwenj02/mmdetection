from xml.dom.minidom import parse
import xml.dom.minidom
import os.path as osp
# 使用minidom解析器打开 XML 文档
root = 'data/erosive2/jingxiu3'
DOMTree = xml.dom.minidom.parse(osp.join(root, 'annotations.xml'))
collection = DOMTree.documentElement 
# 在集合中获取所有电影
images = collection.getElementsByTagName("image")
txt_file = osp.join(root, 'des.txt')

des = {}
with open(txt_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        print(line)
        des[int(line[0])] = line[1]

remove1_txt_f = open(osp.join(root, 'shidao.txt'), 'w')
remove2_txt_f = open(osp.join(root, 'no.txt'), 'w')
remain_all_txt_f = open(osp.join(root, 'fine.txt'), 'w')
remove_txt_f = open(osp.join(root, 'remove.txt'), 'w')

# 打印每部电影的详细信息
for image in images:
    id = int(image.getAttribute("id"))
    name = image.getAttribute("name")
    name = name.split('/')[-1]
    if id in des and '未标注' in des[id]:
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