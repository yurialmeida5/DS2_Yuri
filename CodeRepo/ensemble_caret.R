
data4 <- read_csv("https://raw.githubusercontent.com/cosmin-ticu/DS2_Ensemble-Stacking/main/data/KaggleV2-May-2016.csv")
# some data cleaning
data4 <- select(data4, -one_of(c("PatientId", "AppointmentID", "Neighbourhood"))) %>%
  janitor::clean_names()
# for binary prediction, the target variable must be a factor + generate new variables
data4 <- mutate(
  data4,
  no_show = factor(no_show, levels = c("Yes", "No")),
  handcap = ifelse(handcap > 0, 1, 0),
  across(c(gender, scholarship, hipertension, alcoholism, handcap, diabetes), factor),
  hours_since_scheduled = as.numeric(appointment_day - scheduled_day)
)
# clean up a little bit
data4 <- filter(data4, between(age, 0, 95), hours_since_scheduled >= 0) %>%
  select(-one_of(c("scheduled_day", "appointment_day", "sms_received")))
## Partition dataset again
set.seed(1234)
train_indices_temp <- as.integer(createDataPartition(data4$no_show, 
                                                p = 0.5, list = FALSE))
data_temp <- data4[train_indices_temp, ]
data_test_stacking <- data4[-train_indices_temp, ]
# Further split into 10% training (5% of original) and 90% validation (45% of original)
set.seed(1234)
train_indices_final <- as.integer(createDataPartition(data_temp$no_show, 
                                                p = 0.1, list = FALSE))
data_train_stacking <- data_temp[train_indices_final, ]
data_validation_stacking <- data_temp[-train_indices_final, ]
## Run Caret Ensemble models
set.seed(1234)
trctrlCaretStack <- trainControl(method = "cv",
                                 n = 5,
                                 classProbs = TRUE, # same as probability = TRUE in ranger
                                 summaryFunction = twoClassSummary,
                                 savePredictions = 'all',
                                 index=createFolds(data_train_stacking$no_show, 3))
caretModelList <- caretList(
  no_show~ ., 
  data=data_train_stacking,
  trControl=trctrlCaretStack,
  metric="ROC",
  tuneList=list(
    rf=caretModelSpec(method="rf", family='binomial', tuneGrid=data.frame(.mtry=2)),
    glmnet=caretModelSpec(method="glmnet", family='binomial', tuneGrid=data.frame(.alpha=0 , .lambda=0.01)),
    gbm=caretModelSpec(method="gbm", tuneGrid=data.frame(.n.trees=500, .interaction.depth=2, .shrinkage=0.01, .n.minobsinnode=5))
  )
)
stackedCaretModel <- caretStack(
  caretModelList,
  method='glmnet', # by default
  family = "binomial",
  metric="ROC",
  tuneLength=10,
  trControl=trainControl(
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)
```

```{r, message=FALSE, warning=FALSE, cache=TRUE, include=FALSE}
stackedCaretModel
```

ROC was used to select the optimal model using the largest value. The final values used for the model were alpha = 0.2 and lambda = 0.02400909.

Plotting the stacked model across all the mixing percentage, we see:

```{r, message=FALSE, warning=FALSE, cache=TRUE}
plot(stackedCaretModel)
```

On the training set, this model achieves an AUC of 0.649. Plotting the ROC curve on the validation set, we see:

```{r, message=FALSE, warning=FALSE, cache=TRUE}
## ROC Plot with built-in package 
stacking_pred<-predict(stackedCaretModel, data_validation_stacking, type="prob")
colAUC(stacking_pred, data_validation_stacking$no_show, plotROC = TRUE) # pretty bad but better than individual models
```

The ROC curve is still quite bad, but only very slightly better than the majority of the individual models.

## g. Evaluate ensembles on validation set. Did it improve prediction?

```{r, message=FALSE, warning=FALSE, cache=TRUE}
# g. ----------------------------------------------------------------------
stackedCaretModelRoc_validation <- roc(predictor = predict(stackedCaretModel, 
                                                newdata=data_validation_stacking,
                                                type='prob', decision.values=T), 
                            response = data_validation_stacking$no_show)
stackedCaretModelRoc_validation$auc[1]
```

As expected, the AUC on the validation set is much more lower than the training set. We can see, however, that the stacked model has achieved a higher AUC than most of the other models on the validation set.

```{r, message=FALSE, warning=FALSE, cache=TRUE}
auc_on_validation <- auc_on_validation %>% add_row(model='stacked',AUC=stackedCaretModelRoc_validation$auc[1])
knitr::kable(auc_on_validation)
```

The stacked model has been beaten in performance by the GBM model by the smallest margin possible, a 0.02% difference in AUC.

We can conclude that an ensemble model with 4 models that uses glm as a meta learner improved prediction and AUC is higher than the majority of what we had for any models before.

The performance increase is very slight, however. We can attribute the loss of the stacked model to the GBM model to the difference in packages used. h2o was used for the individual machine learning models and caret was used for the stacked model (however, with the exact same parameters as the best performing h2o models).

For the sake of this assignment, the final model to be evaluated on the test set will be the stacked ensemble model.

## h. Evaluate the best performing model on the test set. How does performance compare to that of the validation set?

```{r, message=FALSE, warning=FALSE, cache=TRUE}
# h. ----------------------------------------------------------------------
stackedCaretModelRoc_test <- roc(predictor = predict(stackedCaretModel, 
                                                           newdata=data_test_stacking,
                                                           type='prob', decision.values=T), 
                                       response = data_test_stacking$no_show)
stackedCaretModelRoc_test$auc[1]