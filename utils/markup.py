'''
author: locchuong,
date: 6/1/22
description: contain the things have iteam1's mark
'''
def banner(path = './iteam1.txt'):
   '''
   print iteam1's banner
   '''
   with open(path,'r') as f:
      t = f.read()
      print(t)
