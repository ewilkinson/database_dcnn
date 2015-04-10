__author__ = 'tigershark'

import xml.etree.ElementTree as ET
import getpass
import os

# username as folder
userName = getpass.getuser()

baseDir = 'configurations' + os.sep + userName + os.sep

# file with db connection details
dbRoot = ET.parse(baseDir + 'db.xml').getroot()

# file with "hard-coded" paths
dirsRoot = ET.parse(baseDir + 'directories.xml').getroot()


def dirs(name):
    return dirsRoot.find(name).text


def db(name):
    return dbRoot.find(name).text