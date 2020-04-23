# naive-bayes-classifier

Using naive bayes classifier to classify newsgroup dataset. The model starts with training of first 500 files in each category where the words and its frequency are stored in a dictionary for easy access to test it in the future . At the end of training model, we will have dictionary of trained files under each category, dictionary of words under each category and a master dictionary of total words. Later, the model is tested with the remaining 500 files.

**Application Used:** Spyder

**Language Used:** Python 3.7

**Dataset:** 20_newsgroups

**Program File:** code.py

**Report Document:** Report_project2.pdf

**Input:**
python code.py

**Output:**

Training the first 500 files

Processing the text.........

Total number of words found in the dataset : 104163

Testing the remaining 500 files

Calculating probabilities.........
Accuracy : 65.0%

