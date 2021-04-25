from xml.dom.minidom import parse
import xml.dom.minidom
 
# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse("data/erosive2/annotations.xml")
collection = DOMTree.documentElement 
# 在集合中获取所有电影
images = collection.getElementsByTagName("image")
txt_file = 'data/erosive2/des.txt'

des = {}
with open(txt_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        print(line)
        des[int(line[0])] = line[1]

remove1_txt_f = open('data/erosive2/shidao.txt', 'w')
remove2_txt_f = open('data/erosive2/no.txt', 'w')
remove_all_txt_f = open('data/erosive2/fine.txt', 'w')
remove_txt_f = open('data/erosive2/remove.txt', 'w')

# 打印每部电影的详细信息
for image in images:
    id = int(image.getAttribute("id"))
    name = image.getAttribute("name")

    if '未标注' in des[id]:
        remove2_txt_f.write(name + '\n')
        remove_txt_f.write(name + '\n')
    if '食管' in des[id]:
        remove1_txt_f.write(name + '\n')
        remove_txt_f.write(name + '\n')
    remove_all_txt_f.write(name.split('/')[-1] + '\n')

remove1_txt_f.close()
remove2_txt_f.close()
remove_txt_f.close()
remove_all_txt_f.close()