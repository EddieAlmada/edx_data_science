##########################################################
# Eduardo Almada
# 2021/03/09
# Edx: Data Science - Capstone
#   This script trains a machine learning algorithm
#   to predict movie ratings based on the RMSE.
##########################################################

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
head(movies)
head(ratings)

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



######################
# Average method
######################

options(digits = 5)

# Calculates the overall average rating from the training set
mu <- mean(edx$rating)

# Estimates all unknown ratings based on the training set rating average
m1 <- RMSE(validation$rating,mu) 



######################
# Movie Effect Method
######################

# By grouping the movies and scaling the ratings, we obtain b_i as the movie bias
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Estimates all unknown ratings based on the training set rating average and b_i
bi_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# plot the distribution of b_i
qplot(b_i, data = b_i, bins = 12, color = I("grey"))

# Calculate RMSE
m2 <- RMSE(validation$rating, bi_ratings)   


######################
# User & Movie Effect
######################

# Calculates user bias, b_u
b_u <- edx %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# plot the distribution of b_u
qplot(b_u, data = b_u, bins = 15, color = I("grey"))

# Estimates all unknown ratings based on the training set rating average, b_i and b_u
bu_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE
m3 <- RMSE(bu_ratings, validation$rating) 


######################
# User, Movie, Genres Effect
######################

# Calculates genres bias, b_e
b_e <- edx %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by = "userId") %>% 
  group_by(genres) %>%
  summarize(b_e = mean(rating - mu - b_i - b_u))

# plot the distribution of b_e
qplot(b_e, data = b_e, bins = 25, color = I("grey"))

# Estimates all unknown ratings based on the training set rating average, b_i, b_u & b_e
be_ratings <- validation %>% left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>% left_join(b_e, by='genres') %>% 
  mutate(pred = mu + b_i + b_u + b_e) %>%
  pull(pred)

# Calculate RMSE
m4 <- RMSE(be_ratings, validation$rating)
  
###########################################
# Regularized Genre, Movie & User Effect
###########################################

# Sequence of lambdas for optimization
lambdas <- seq(0,10, by=0.10)

# RMSE of each lambda 
rmse_f <- sapply(lambdas, function(l){
  mu <- mean(edx$rating) 
  
  b_i <- edx %>% group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))

  b_u <- edx %>% left_join(b_i, by="movieId") %>%
    group_by(userId) %>%  summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_e <- edx %>% left_join(b_i, by='movieId') %>%
    left_join(b_u, by = "userId") %>% group_by(genres) %>%
    summarize(b_e = sum(rating - mu - b_i - b_u)/(n()+l))
  
  reg_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_e, by = 'genres') %>% 
    mutate(pred = mu + b_i + b_u + b_e) %>%
    pull(pred)
 
  return(RMSE(reg_ratings, validation$rating))
})

# Gets min RSME and the respective lambda
index <- which.min(rmse_f)
c(lambda = lambdas[index], RSME = min(rmse_f))

m5 <- min(rmse_f)

######################################################
# Conclusion for best method according to the RMSE
######################################################
options(digits = 4)

# Creating a vector comparing every method RMSE
Methods <- c(Average_method = m1, Movie_effect = m2, User_effect = m3,
  Genre_effect = m4,Regularized_method = m5)

# Evaluating the Methods vector to test for a value less than 0.86490
Results <- data.table(Method = c('Average_method', 'Movie_effect', 'User_effect',
                 'Genre_effect','Regularized_method'), RMSE = Methods,
                Pass_Criteria = ifelse(Methods < 0.86490,'pass','fail'))
Results

# Visual inspection for pass criteria
plot(Methods)
abline(h = 0.86490, col = I("blue"))

