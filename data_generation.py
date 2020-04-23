# import itertools
import re
import random
# c = itertools.combinations(warren_list, 4)
filename = 'C:/Users/shubham-naik/Documents/warren-time-new/ngpd-merceros-warren-nlp-be/data/examples/chatito/entity/combined.chatito'
with open(filename, 'r') as f:
    l = f.read()
    
    
lines = l.split('\n')
# lines = ['~[warren_prefix]']
for line in lines:
    words = line.split(' ')
    words = [word for word in words if word!='']
    print('words->', words)
    for i in range(words):
        word = words[i]
        if word.startswith('~') :
            print('~~~~~~~~~~~~~~~')
            search_str = word.split('[')[1].split(']')[0]
            search_str = '~['+search_str+']'
#             search_str2 = '~['+search_str+'?]'
            if '?' in search_str:
                search_str = search_str.replace('?','')
            print('collection item ->', search_str)
            c = 0
            for line_new in lines:
                c += 1
#                 if not len(line_new.split(' ')) >1:
                
                list_item = []
                if not re.match(r'\s', line_new) and (search_str in line_new): # or search_str2 in line_new):
#                     print(line_new, c)
                    for lin in lines[c:]:
                        if lin == "":
                            break
                        list_item.append(lin.strip())
                    print('list_item-->', list_item)
        words[i] = random.choice(list_item)
        if word.startswith('@'):
            print('@@@@@@@@@@@@@@@@@@@@@')
            search_str = word.split('[')[1].split(']')[0]
            search_str = '@['+search_str+']'
            print('entity-->>', search_str)
            c =0 
            for line_new in lines:
                c+=1 
                list_item = []
                if not re.match(r'\s', line_new) and search_str in line_new:# or not line_new.startswith('%'):
#                     print(line_new, c)
                    for lin in lines[c:]:
                        if lin == "":
                            break
                        list_item.append(lin.strip())
                    print('entity_item-->', list_item)
                
        

                    
#             print(search_str)
    
            
    
#     if line.startswith('%') or line != "\n":
#     if line == "":
# #             print('insode break')
#         break
# #     print(line)
#     if line.startswith('~'):
#         search_str = line
#         for
        
