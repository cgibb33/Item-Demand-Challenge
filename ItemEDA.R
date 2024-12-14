library(tidymodels)
library(tidyverse)
library(dplyr)
library(forecast)
library(gridExtra)
library(ranger)
library(dbarts)
library(parsnip)
library(vroom)

##Load in Data Sets
IDtrain <- vroom('train.csv')
IDtest <- vroom('test.csv')

# Filter down to just 1 store item for exploration and model building
storeItem1 <- IDtrain %>%
filter(store==1, item==1)

tsp1 <- storeItem1 %>%
ggplot(mapping=aes(x=date, y=sales)) +
geom_line() +
geom_smooth(se=FALSE)

ACF1a <- storeItem1 %>%
pull(sales)%>%
forecast::ggAcf(.)

ACF1b <- storeItem1 %>%
pull(sales)%>%
forecast::ggAcf(., lag.max=2*365)

storeItem2 <- IDtrain %>%
  filter(store==1, item==2)

tsp2 <- storeItem2 %>%
  ggplot(mapping=aes(x=date, y=sales)) +
  geom_line() +
  geom_smooth(se=FALSE)

ACF2a <- storeItem2 %>%
  pull(sales)%>%
  forecast::ggAcf(.)

ACF2b <- storeItem2 %>%
  pull(sales)%>%
  forecast::ggAcf(., lag.max=2*365)

grid.arrange(tsp1, ACF1a, ACF1b, tsp2, ACF2a, ACF2b, nrow = 2, ncol = 3)

#Create Recipe

Timerecipe <- recipe(sales ~ ., data = IDtrain) %>%
  step_date(date, features = "doy") %>%
  step_date(date, features = "dow") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy)) %>%
  step_dummy(date_dow) %>%  # Convert 'date_dow' to dummy variables
  step_rm(date)  # Remove the original 'date' column

prep <- prep(Timerecipe)
bake(prep, IDtrain)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 250
) %>%
set_engine("keras") %>% #verbose = 0 prints off less
  set_mode("regression")

nn_wf <- workflow() %>%
  add_recipe(Timerecipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 6)),
                            levels=5)

folds <- vfold_cv(IDtrain, v = 5, repeats=1)

tuned_nn <- nn_wf %>%
tune_grid(resamples = folds,
          grid = nn_tuneGrid,
          metrics = metric_set(smape))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="smape") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## Find Best Tuning Parameters
bestTune <- tuned_nn %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=IDtrain)

## Predict
nnfinal <- final_wf %>%
  predict(new_data = IDtest, type="numeric")

nn_submission <- nnfinal %>%
  bind_cols(IDtest) %>%
  select(id, )  # Select 'id' and 'ACTION' columns for the submission

# Write the submission file
vroom_write(x = penlog_submission, file = "./amazon_penlog.csv", delim = ",")
