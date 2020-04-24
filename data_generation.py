## Run the script using below command
## python .\generate_data.py --chatito_file path to chatito file

import re
import random
import argparse

class GenData:
    def __init__(self, filename):
        self.filename = filename
        self.ent_dict = {}
        self.coll_dict = {}
        self.md_dict = {}
        with open(self.filename, 'r') as f:
            l = f.read()
            self.lines = l.split('\n')

    def generate_lists(self, word, lines):
        if word.startswith('~'):
            search_str = word.split('[')[1].split(']')[0]
            search_str = '~['+search_str+']'
            if '?' in search_str:
                search_str = search_str.replace('?','')
            c = 0
            for line_new in lines:
                c += 1
                list_item = []
                if not re.match(r'\s', line_new) and (search_str in line_new): # or search_str2 in line_new):
                    for lin in lines[c:]:
                        if lin == "":
                            break
                        list_item.append(lin.strip())
                    break
        if word.startswith('@'):
            search_str = word.split('[')[1].split(']')[0]
            search_str = '@['+search_str+']'
            c =0 
            for line_new in lines:
                c+=1 
                list_item = []
                if not re.match(r'\s', line_new) and search_str in line_new:# or not line_new.startswith('%'):
                    for lin in lines[c:]:
                        if lin == "":
                            break
                        list_item.append(lin.strip())
                    break
        return list_item

    def store_lists(self):
        for line in self.lines:
            words = line.split(' ')
            words = [word for word in words if word!='']
            intent = [word for word in words if word.startswith('%')]
            entities = [word for word in words if word.startswith('@')]
            collections = [word for word in words if word.startswith('~')]

            for word in entities:
                if not word in self.ent_dict.keys():
                    self.ent_dict[word] = self.generate_lists(word, self.lines)

            for word in collections:
                if not word in self.coll_dict.keys():
                    self.coll_dict[word] = self.generate_lists(word, self.lines)
            
    def process(self):
        lc = 0
        for line in self.lines:
            lc += 1
            if line.startswith('%'):
                line_cpy = line[:]
                line_cpy = line_cpy.replace("%", '')
                line_cpy = line_cpy.replace("[", '')
                intent = line_cpy.replace("]", '')
                intent = "## intent:"+intent
                print('intent->', intent)
                self.md_dict[intent] = []
            next_line = self.lines[lc]
            if line == "" and '%' not in next_line:
                break
            org_words = line.split(' ')
            org_words = [word for word in org_words if word!='']
            entities = [word for word in org_words if word.startswith('@')]
            collections = [word for word in org_words if word.startswith('~')]
            if len(entities) != 0:
                for entity in entities:
                    ent_val = entity.split('[')[1].split(']')[0]
                    ent = self.ent_dict[entity]
                    for e in ent:
                        words = org_words[:]
                        temp_words = org_words[:]
                        for j in range(len(words)):
                            word = words[j]
                            if word.startswith('~') and '?' in word:
                                choice = random.choice([0,1])
                                if choice == 0:
                                    w = random.choice(self.coll_dict[word])
                                    words[j] = w
                                else:
                                    words[j] = ''
                            if word.startswith('@') and word == entity:
                                words[j] = '['+e+']('+ent_val+')'
                            if word.startswith('@') and word != entity:
                                word_ent = word.split('[')[1].split(']')[0]
                                words[j] = '['+random.choice(self.ent_dict[word])+']('+word_ent+')'
                            if word.startswith('~') and '?' not in word:
                                w = random.choice(self.coll_dict[word])
                                words[j] = w
                        words = [word for word in words if word !='']
                        final_text = ' '.join(words)
                        self.md_dict[intent].append('- '+final_text)
            if len(collections) != 0 and len(entities) == 0:
                v = [len(self.coll_dict[c]) for c in collections]
                col = collections[v.index(max(v))]
                col_values = self.coll_dict[col]
                for col_val in col_values:
                    words = org_words[:]
                    for j in range(len(org_words)):
                        word = words[j]
                        if word.startswith('~') and '?' in word:
                            choice = random.choice([0,1])
                            if choice == 0:
                                w = random.choice(self.coll_dict[word])
                                words[j] = w
                            else:
                                words[j] = ''
                        if word.startswith('~') and word == col:
                            words[j] = col_val
                        if word.startswith('~') and '?' not in word:
                            w = random.choice(self.coll_dict[word])
                            words[j] = w
                    words = [word for word in words if word !='']
                    final_text = ' '.join(words)
                    self.md_dict[intent].append('- '+final_text)
            if '%' not in line and len(entities) == 0 and len(collections)==0 and line !='':
                final_text = ' '.join(org_words)
                self.md_dict[intent].append('- '+final_text)
                
        return self.md_dict
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python program to generate the training data.')
    parser.add_argument('--chatito_file', help='chatito file', type=str)
    args = parser.parse_args()
    filename = args.chatito_file
    obj = GenData(filename)
    obj.store_lists()
    md_dict = obj.process()
    final_dict = {}
    for k, v in md_dict.items():
        final_dict[k] = set(v)
    print(final_dict.keys())
    with open('nlu.md', 'w') as f:
        for k,v in final_dict.items():
            f.write(k)
            f.write('\n')
            for val in v:
                f.write(val)
                f.write('\n')
            f.write('\n')
