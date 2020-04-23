# Mohan, Shreyas
# 1001-669-806
# 2019-10-23
# Project 2
import os
import copy,math

#root_path refers to the path of the dataset
root_path = '20_newsgroups/'
list_of_folders = os.listdir(root_path)# returns a list containing the names of the categories in the newsgroups directory
train_data = 500
master_dictionary = {} #dictionary of trained files under each category

master_word_freq = {} #a master dictionary of total words.
category_dict = {} #dictionary of words under each category 
category = 20
alpha = 0.0001 #laplace smoothing

#Filtering the data
def data_handler(text):
    spl_characters = ['~','`','!','@','^','$','%','&','*','(',')','+','=','{','}','[',']',';',':','|','\\','"',"'",'\n','<','>',',','.','?','/','-','*']
    for i in spl_characters:
        text = text.replace(i , ' ')
    text = text.lower()
    #stop words are eliminated for more accurate prediction
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for j  in stop_words:
        text = text.replace(j, '')
    return text

print('Training the first 500 files')
print('Processing the text.........')
#Navigate through each category
for folder in list_of_folders:
    mapper = {}#dic
    count = 0
    list_of_files = os.listdir(root_path + folder)#List of files in each category
    for file in list_of_files:
        if count < train_data:
            file_path = root_path + folder + '/' + file
            curr_file = open(file_path, 'r')
            #Read the file
            file_data = data_handler(curr_file.read())
            for word in file_data.split(' '):
                word = word.strip()
                if word != ' ' and word != '':
                    if mapper.get(word,0) == 0:
                        mapper[word] = 1
                    else:
                        #increase the frequency count of word
                        mapper[word] = mapper.get(word,0) + 1
                    if master_word_freq.get(word,0) == 0:
                        master_word_freq[word] = 1
                    else:
                        master_word_freq[word] = master_word_freq.get(word,0) + 1
            list_of_files.remove(file)
        count= count+1
    master_dictionary[folder] = list_of_files
    category_dict[folder] = mapper
print('Total number of words found in the dataset : ' + str(len(master_word_freq)))
print('Testing the remaining 500 files')
test_folders = copy.deepcopy(list_of_folders)
probabilities = []
res_dir = {}
j=0
print('Calculating probabilities.........')
for folder in test_folders:
    res_dir[folder] = 0
for folder in test_folders:
    t_list_of_files = os.listdir(root_path + folder)
    flag =0
    prob1 = 0.0
    for file in t_list_of_files:
        if file not in master_dictionary[folder]:
            flag+=1
            file_path = root_path + folder + '/' + file
            curr_file = open(file_path, 'r')
            file_data = data_handler(curr_file.read())
            sum1 = sum(category_dict[folder].values())
            for word in file_data.split(' '):
                if word != ' ' and word != '' and word not in category_dict[folder].keys():
                    word = word.strip()
                    prob2 = float(category_dict[folder].get(word, 0.0)) + alpha
                    prob1 = prob1 +(math.log(float(prob2)/float(sum1)) )
                    
            t_list_of_files.remove(file)
    probabilities.append(prob1)
    res_dir[test_folders[probabilities.index(max(probabilities))]] += 1
frequency = list(res_dir.values())
maxi = max(frequency)
accuracy = maxi/category*100
print("Accuracy : " + str(accuracy) + "%")    