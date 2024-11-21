## Preprocessing done by previous task (by Candis),
## Detail can be found "scripts/prelim_task1_candis.Rmd"
## here we load the dataset directly

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
load("data/cleaned_claims_headers.RData")

## Select the subset of useful information
claims_clean<- parse_data_headers(claims_raw)%>%
  select(.id, bclass, mclass, text_clean)


# Tokenization first 
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

# one-hot encoding
# Encode binary label
claims_clean <- claims_clean %>% 
  mutate(y_binary = to_categorical(as.numeric(as.factor(claims_clean$bclass))-1))
# Encode multiclass label
claims_clean <- claims_clean %>% 
  mutate(y_multiclass = to_categorical(as.numeric(as.factor(claims_clean$mclass))-1))

# Train-test split
set.seed(1)
partition <- initial_split(claims_clean,prop = 0.8)  # 80% training, 20% testing
train_data <- training(partition)
test_data <- testing(partition)

X_train <- train_data$sequences
X_test <- test_data$sequences

# Binary label
y_train_binary <- train_data$y_binary
y_test_binary <- test_data$y_binary

# Multiclass label
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

# save the binary model
save_model_tf(binary_model, "results/RNN_models/RNN-Binary-Model.RData")
# save the multiclass model
save_model_tf(multi_model, "results/RNN_models/RNN-Multiclass-Model.RData")
