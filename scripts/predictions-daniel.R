library(tidyverse)
library(tidymodels)
library(tidytext)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)
library(dplyr)
library(yardstick)
library(tidyr)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
load('data/claims-test.Rdata')
# path to activity files on repo
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
# load functions needed for this assignment
source(paste(url, 'projection-functions.R', sep = ''))

# Load saved models and objects
fit_reg <- readRDS("results/models/binary_model.rds")
fit_reg_multi <- readRDS("results/models/multiclass_model.rds")
proj_out <- readRDS("results/models/variables/proj_out.rds")
terms <- readRDS("results/models/variables/terms.rds")
lambda_opt <- readRDS("results/models/variables/lambda_opt.rds")
lambda_multi_opt <- readRDS("results/models/variables/lambda_multi_opt.rds")




# Pre-processing for 'claims-test.Rdata' using headers
# From Candis.Rmd
# function to parse html and clean text
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


# apply preprocessing pipeline
test_clean_df <- claims_test %>%
  parse_data_headers() %>%
  select(.id, text_clean)

# Tokenize and lemmatize
tfidf_test_claim <- test_clean_df %>% 
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words',
                stopwords = str_remove_all(stop_words$word, '[[:punct:]]')) %>%
  mutate(token.lem = lemmatize_words(token)) %>%
  filter(str_length(token.lem) > 2) %>%
  count(.id, token.lem, name = 'n') %>%
  bind_tf_idf(term = token.lem, 
              document = .id,
              n = n) %>%
  pivot_wider(id_cols = c(.id),
              names_from = 'token.lem',
              values_from = 'tf_idf',
              values_fill = 0)

# Create a full term list
all_terms <- c(".id", terms)
missing_terms <- setdiff(terms, names(tfidf_test_claim))

# Add missing terms with zeros
if (length(missing_terms) > 0) {
  tfidf_test_claim[missing_terms] <- 0
}

# Ensure columns are in the same order
test_dtm <- tfidf_test_claim %>%
  select(.id, all_of(terms))

# Store .id separately
test_ids <- test_dtm$.id

# Remove .id from test_dtm
test_dtm <- test_dtm %>% select(-.id)


# Project test data DTM onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

# Coerce projected test data to matrix
x_test <- as.matrix(test_dtm_projected)

# Binary Classification Predictions
# Compute predicted probabilities
preds_binary <- predict(fit_reg, 
                        s = lambda_opt, 
                        newx = x_test,
                        type = 'response')

# Convert probabilities to binary labels
bclass_levels <- fit_reg$classnames  # Extracts class names from the glmnet model
bclass.pred <- factor(ifelse(preds_binary > 0.5, bclass_levels[2], bclass_levels[1]),
                      levels = bclass_levels)


# Multinomial Classification Predictions
# Compute predicted probabilities for multiclass
preds_multi <- predict(fit_reg_multi, 
                       s = lambda_multi_opt, 
                       newx = x_test,
                       type = 'response')

# Convert probabilities to class labels
mclass_levels <- fit_reg_multi$classnames

# Get the predicted class labels
pred_class <- apply(preds_multi[,,1], 1, function(x) mclass_levels[which.max(x)])
mclass.pred <- factor(pred_class, levels = mclass_levels)


# Create pred_df
pred_df <- tibble(
  .id = test_ids,
  bclass.pred = bclass.pred,
  mclass.pred = mclass.pred
)

save(pred_df, file = "results/preds-group7.RData")
