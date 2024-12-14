library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)

# Read in Data----------------------------
train <- vroom("/kaggle/input/demand-forecasting-kernels-only/train.csv")
test <- vroom("/kaggle/input/demand-forecasting-kernels-only/test.csv")


nStores <- max(train$store)
nItems <- max(train$item)

for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)
    
    my_recipe <- recipe(sales~., data=train) %>%
      step_date(date, features = c("dow", "month", "decimal", "doy", "year")) %>% 
      step_holiday(date, role = "predictor") %>% 
      step_range(date_doy, min = 0, max = pi) %>%                                 
      step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy)) %>%             
      step_lencode_mixed(all_nominal_predictors(), outcome = vars(sales)) %>%     
      step_normalize(all_numeric_predictors()) %>%                                
      step_rm(date, item, store)   
    
    lgbm_model <- boost_tree(tree_depth=2, 
                             trees=1000,
                             learn_rate=0.01) %>%
      set_engine("lightgbm") %>%
      set_mode("regression")
    
    boost_wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(lgbm_model)
    
    fitted_wf <- boost_wf %>%
      fit(data=storeItemTrain)
    
    preds <- predict(fitted_wf, new_data=storeItemTest) %>%
      bind_cols(storeItemTest) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

vroom_write(all_preds, file="submission.csv", delim=",")