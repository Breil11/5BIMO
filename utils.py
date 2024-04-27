import os

def ReadFolder(basename):
    contents = os.listdir(basename)
    return contents

def ReadFolderIndex(contents,index):
    return contents[index]

