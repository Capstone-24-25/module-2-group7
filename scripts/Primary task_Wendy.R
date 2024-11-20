library(keras)
library(text2vec)
library(tidyverse)
library(tidymodels)
library(tidytext)
library(tensorflow)
library(dplyr)
library(SnowballC)
library(tm)
library(caret)
library(textclean)
library(glmnet)
library(modelr)
library(Matrix)
library(sparsesvd)
library(yardstick)


## Pre-trained model

# Load data
load('data/claims-raw.RData')
## Preprocessing with header from Candis
parse_fn_headers <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>% # remove url
    rm_email() %>% # remove email
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>% # lowercased
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data_headers <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_headers(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}


claims_clean<- parse_data_headers(claims_raw)%>%
  select(.id, bclass, mclass, text_clean)


# Tokenization
tokenizer <- text_tokenizer(num_words = 5000)
tokenizer %>% fit_text_tokenizer(claims_clean$text_clean)

# Convert text to sequences
sequences <- texts_to_sequences(tokenizer, claims_clean$text_clean)
# Set sequence length
maxlen <- 100 
# Set equal length
same_len_data <- pad_sequences(sequences, maxlen = maxlen)
claims_clean <- claims_clean %>% 
  mutate(sequences = same_len_data)

# Encode binary label
claims_clean <- claims_clean %>% 
  mutate(y_binary = to_categorical(as.numeric(as.factor(claims_clean$bclass))-1))
# Encode multiclass label
claims_clean <- claims_clean %>% 
  mutate(y_multiclass = to_categorical(as.numeric(as.factor(claims_clean$mclass))-1))

# Train-test split
set.seed(1)
partition <- initial_split(claims_clean, prop = 0.8)  # 80% training, 20% testing
train_data <- training(partition)
test_data <- testing(partition)

X_train <- train_data$sequences
X_test <- test_data$sequences

#Binary label
y_train_binary <- train_data$y_binary
y_test_binary <- test_data$y_binary

#Multiclass label
y_train_multiclass <- train_data$y_multiclass
y_test_multiclass <- test_data$y_multiclass


# Build RNN model for binary classification
binary_model <- keras_model_sequential()
binary_model %>%
  layer_embedding(input_dim = 5000, output_dim = 128, input_length = maxlen) %>%  # Embedding layer
  layer_simple_rnn(units = 64, return_sequences = FALSE) %>%  # RNN layer
  layer_dense(units = 2) %>%  # Dense layer
  layer_activation(activation = 'sigmoid')  # Activation layer

summary(binary_model)

binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('binary_accuracy')
)


# Train the binary classification model
history_binary <- binary_model %>% fit(
  X_train, y_train_binary,
  epochs = 10,
  batch_size = 32
)


binary_pred<-predict(binary_model, X_test)
bclass_pred <- factor(ifelse(binary_pred[,2] > 0.5, 1, 0), levels = c(0, 1))

conf_matrix_binary <- confusionMatrix(bclass_pred, as.factor(as.numeric(test_data$bclass)-1))
conf_matrix_binary$overall["Accuracy"]


# Build RNN model for multi-clss classification
multi_model <- keras_model_sequential()
multi_model %>%
  layer_embedding(input_dim = 5000, output_dim = 128, input_length = maxlen) %>%  # Embedding layer
  layer_simple_rnn(units = 64, return_sequences = FALSE) %>%  # RNN layer
  layer_dense(units = 5) %>%  # Dense layer
  layer_activation(activation = 'softmax')  # Activation layer

summary(multi_model)

multi_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


# Train the multi class classification model
history_multi <- multi_model %>% fit(
  X_train, y_train_multiclass,
  epochs = 10,
  batch_size = 32
)

multi_pred<-predict(multi_model, X_test)
mclass_pred <- apply(multi_pred, 1, which.max) - 1
conf_matrix_multi <- confusionMatrix(as.factor(mclass_pred), as.factor(as.numeric(test_data$mclass)-1))
conf_matrix_multi$overall["Accuracy"]

evaluate()

pred_df_model <- data.frame(
  .id = test_data$.id, 
  bclass.pred = bclass_pred,
  mclass.pred = mclass_pred
)

pred_df_model$bclass.pred <- factor(pred_df_model$bclass.pred, 
                              levels = c(0,1), 
                              labels =c("N/A: No relevant content.", "Relevant claim content"))

pred_df_model$mclass.pred <- factor(pred_df_model$mclass.pred, 
                              levels = c(0,1,2,3,4), 
                              labels =c("N/A: No relevant content.", 
                                        "Physical Activity",
                                        "Possible Fatality",
                                        "Potentially unlawful activity",
                                        "Other claim content"))
# pred_df_model

######################## Test on claim test 
load("data/claims-test.RData")
## Preprocessing

parsed_claim_test <- parse_data_headers(claims_test)

# Tokenization
tokenizer_test <- text_tokenizer(num_words = 5000)
tokenizer_test %>% fit_text_tokenizer(parsed_claim_test$text_clean)

# Convert text to sequences
sequences_test <- texts_to_sequences(tokenizer_test, parsed_claim_test$text_clean)
# Set sequence length
maxlen <- 100 
# Set equal length
same_len_data_test <- pad_sequences(sequences_test, maxlen = maxlen)
parsed_claim_test <- parsed_claim_test %>% 
  mutate(sequences_test = same_len_data_test)


binary_pred_test <- predict(binary_model, parsed_claim_test$sequences_test)
bclass_pred_test <- factor(ifelse(binary_pred_test[,2] > 0.5, 1, 0), levels = c(0, 1))


multi_pred_test<-predict(multi_model, parsed_claim_test$sequences_test)
mclass_pred_test <- apply(multi_pred_test, 1, which.max) - 1

pred_df <- data.frame(
  .id = parsed_claim_test$.id, 
  bclass.pred = bclass_pred_test,
  mclass.pred = mclass_pred_test
)

pred_df$bclass.pred <- factor(pred_df$bclass.pred, 
                              levels = c(0,1), 
                              labels =c("N/A: No relevant content.", "Relevant claim content"))

pred_df$mclass.pred <- factor(pred_df$mclass.pred, 
                              levels = c(0,1,2,3,4), 
                              labels =c("N/A: No relevant content.", 
                                        "Physical Activity",
                                        "Possible Fatality",
                                        "Potentially unlawful activity",
                                        "Other claim content"))
pred_df
