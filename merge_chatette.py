import os

chatette_path = './examples/chatito/entity_chatette/'

chatette_files = [file for file in os.listdir(chatette_path) if file.endswith('.chatette')]
c=0
for file in chatette_files:
    print('file->', file)
    c += 1
    os.system('python -m chatette ./examples/chatito/entity_chatette/'+file+' -o ./examples/chatete_output/'+str(c)+' -a rasa-md')
