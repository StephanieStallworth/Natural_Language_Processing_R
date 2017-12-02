# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote ='', stringsAsFactors = FALSE)

# Cleaning the texts 
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review)) # create corpus
corpus = tm_map(corpus, content_transformer(tolower)) # transform words to lowercase
corpus = tm_map(corpus, removeNumbers) # remove numbers
corpus = tm_map(corpus, removePunctuation) # remove punctuation
corpus = tm_map(corpus, removeWords, stopwords()) # remove stopwords
corpus = tm_map(corpus, stemDocument) # stem to root word
corpus = tm_map(corpus, stripWhitespace)


# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)# only keep this percentage of words that are the most frequent


# Transform sparse matrix into dataframe of both dependent and independent variables
dataset = as.data.frame(as.matrix(dtm)) # independent variables only
dataset$Liked = dataset_original$Liked # add dependent variable

# Training Random Forest Classification
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, # dependent variable
                       levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, # dependent variable
                     SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Only have 0 and 1 in the spare matrix of features
# therefore we don't have an indepenent variable dominating another independent variable
# thus, feature scaling is not needed here
# training_set[-3] = scale(training_set[-3])
# test_set[-3] = scale(test_set[-3])

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-692], # everthing but index of dependent variable
                          y = training_set$Liked, # dependent variable 
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692]) # test test excluding index of dependent variable

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)

# Accuracy Rate
accuracy = (78 + 72)/(78+22+28+72)
print(accuracy)
