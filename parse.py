# Importing the required libraries
import xml.etree.ElementTree as Xet
import pandas as pd
  
cols = ["filename", "width", "height", "depth", "segmented", "obj"]
rows = []
  
# Parsing the XML file
xmlparse = Xet.parse('./train/apple_3.xml')
root = xmlparse.getroot()
filename = root.find("filename").text
size = root.find('size')
width = size.find("width").text
height = size.find("height").text
depth = size.find("depth").text
segmented = root.find("segmented").text

obj_list = []
for obj in root.findall('object'):
    name = obj.find('name').text
    truncated = obj.find('truncated').text
    difficult = obj.find('difficult').text
    bndbox = obj.find('bndbox')
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
    list = [name, truncated, difficult, xmin, ymin, xmax, ymax]
    obj_list.append('-'.join(list))

rows.append({"filename": filename,
                "width": width,
                "height": height,
                "depth": depth,
                "segmented": segmented,
                "obj": '#'.join(obj_list)})
  
df = pd.DataFrame(rows, columns=cols)
  
# Writing dataframe to csv 
df.to_csv('output.csv')