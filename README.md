# Goodreads Books Dataset Analysis
NAME | DATE
------------ | -------------
 OZGE GUNER | March 27, 2020

### Resources
* Python code for the analysis : __project_code.py__
* Figures : __plots__

### Research Question
Is it possible to guess whether an author’s newly published book is going to be reader’s favourite book or not by taking book’s average number of pages and author's books ratings count into account?
### Abstract
In this analysis, we examine how the number of pages of a book and popularity (represents ratings count) of an author affect book’s average rating through analysing author’s published books year dependence. By dividing the publication years into two as before and after 2003, we classified dataset while adding new information related to publication year such as showing whether new books have more popularity counts than old books or not for each author. Using these together with author’s average number of pages, we aim to predict whether readers will love newly published or upcoming book of a specific author or not by considering its number of pages and author's previous ratings. Applying machine learning algorithm for this dataset resulted in not very high accuracies but nevertheless can be promising strategy to help publishers, book sellers and even readers to guide them as recommendation engine. 
### Introduction
The dataset that is used for the analysis was scraped via the Goodreads API which is publicly available in Kaggle.(https://www.kaggle.com/jealousleopard/goodreadsbooks). It includes detailed information about books in different languages such as authors, title of the books, ratings count, text reviews count, average ratings, publication years,  publishers, the number of page of books, isbn (unique number to identify the book, the International Standard Book Number), isbn13 (A 13-digit isbn), bookID (A unique identification number of each book). For our research, we use a subset of this dataset by eliminating some features such as isbn13, isbn, languages, titles of the books, publishers, text reviews count and bookID.
### Methods
We analyze this dataset firstly by removing outliers (ratings_count < 900000 and page_number < 2000) and rows where ratings_counts is 0 since we develop our data structure based on popularity. Secondly, by taking their publication years into account, which we labeled the data as ‘new’ if the book was published after 2003 or ‘old’ if it was published before 2003. Then, we group dataset consisting of only ‘new’ books by ‘authors’ while taking mean of corresponding average_rating, ratings_count, and page_number columns. We do exactly same thing but using dataset consisting of ‘old’ books this time . Finally we compare ratings count of ‘new’ books and ‘old’ books, and then we add a new categorized column that shows ’new’ if there is more ‘new’ books ratings than old books ratings, or ‘old' for vice versa. We also categorize the target (average_ratings), which is renamed by ‘favourite’, and we labeled it as ‘yes’ if the author's average ratings is in range (4,5) or ‘no’ else. For the machine learning process, categorical variables are converted into dummy variables.  
For this classification problem, we check the correlation between the features and target (favourite). As it seem from the figure below that correlations between features and output are weak, which therefore require a strong optimization algorithm to determine best decision boundaries.

![](https://github.com/ozggnr/BigDataProject/blob/master/plots/correlation_heatmap.png)

 In this sense, the method that we used for modelling is one of the ensemble methods called Adaptive Boosting Classifier (AdaBoostClassifier). It can build a strong classifier by starting from the combination of multiple poorly performing classifiers (worse than random prediction), takes different subsets of training sets into account, tries to reduce the error by selecting worst subset (by assigning higher weight to it) and then adjust the weights (both classifiers and training subsets) iteratively until it reaches to the optimum, which can help us to get high accuracy score. Here, we used Decision Tree Classifier as base estimator (by default), estimator size as 100 and learning rate as 1.3, which were optimized based on the accuracy score.
### Results
Best accuracy score we obtained using AdaBoostClassifier with the parameters above was 0.63, which indicates that we are better than random predictions. To understand how many right and false corrections that our classifier has made was demonstrated using confusion matrix below. 

   ![](https://github.com/ozggnr/BigDataProject/blob/master/plots/confusionmatrix_heatmap.png)

We calculated that recall, which defines how corresponding class is correctly recognized, and we found that it is 0.88 for 0 (not favourite books) and 0.26 for 1 (favourite books). It indicates that the problem is the prediction of favourite books, which looks it is not correctly recognized by the classifier. On the other hand, precision, which defines how accurately our class is predicted, was found as 0.64 for not favourite and 0.62 for favourite, which indicates that precision values are not helping us to understand why predictions fail unlike the values we obtained for recall.

### Discussion
Our method does not exactly solve our problem since 0.63 accuracy score is not a very high prediction rate to claim that we can predict whether any author’s newly published or upcoming book will be loved or not by just taking into account of this books’s number of pages and book ratings counts of the authors (‘new’ or ‘old’). 

![](https://github.com/ozggnr/BigDataProject/blob/master/plots/favourite_book_distribution.png)

As it seen in the figure above, number of pages are concentrated between 0 and 700. Moreover, favourite book distribution in terms of ‘yes’ and ‘no’ in number of pages vs book ratings count indicate that these labels are not distributed distinctively, which weakens our model. Therefore, to improve the model, we can suggest that more data should be collected in the range of 0 and 700 for number of pages together with book rating counts, publication years, which may lead to appearance of more distinctive ‘yes’ and ‘no’ clusters. In addition to this, the new features can be added such as a column include the information about books which the readers want to read, currently reading or already read. This information already exists in the Goodreads website.
### References
1. https://www.kaggle.com/jealousleopard/goodreadsbooks
2. https://www.goodreads.com



