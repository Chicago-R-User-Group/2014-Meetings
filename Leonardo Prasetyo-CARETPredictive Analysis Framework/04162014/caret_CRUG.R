
## ----echo=FALSE, results='hide',message=FALSE----------------------------
options(rstudio.markdownToHTML = 
  function(inputFile, outputFile) {      
    require(markdown)
    markdownToHTML(inputFile, outputFile, stylesheet='custom.css')   
  }
)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
library(caret)
train_idx <- createDataPartition(iris$Species,
                                  # 80% goes to training
                                  p = .8,     
                                  list = FALSE,
                                  times = 1)
head(train_idx)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
iris_train <- iris[ train_idx,]
nrow(iris_train)
iris_test  <- iris[-train_idx,]
nrow(iris_test)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
# create 10-fold cross validated stratified 
cv_idx <- createFolds(iris_train$Species, 10)
str(cv_idx)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
cv_ts <- createTimeSlices(1:12, 3, 1, fixedWindow = FALSE)
str(cv_ts[[1]])


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
wine <- read.csv("./wineTrain.csv")
test_wine <- read.csv("./wineTest.csv", as.is = T)
str(wine)
error.rate <- function(x, ref, digs = 3) round(mean(x != ref), digs) 


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
wine$qual5 <- cut(wine$quality, c(3, 5:8, 10) - 0.5,
                 labels = c("Very Low", "Low", "Average", "High",
                            "Very High"))

test_wine$qual5 <- cut(test_wine$quality, c(3,5:8,10)-0.5, 
                      labels=c("Very Low","Low","Average","High",
                               "Very High"))


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
# scale the wine dataset for kNN
norm_wine <- as.data.frame(cbind(scale(wine[, c(-12,-13)]), 
                                 qual5=wine[, 13]))

# similar scaling for test wine dataset using the same moments
# as the training set
norm_test_wine <- as.data.frame(
  cbind(scale(test_wine[, c(-12,-13)],
              center=apply(wine[, c(-12,-13)], 2, mean), 
              scale=apply(wine[, c(-12, -13)], 2, sd)),
              qual5=test_wine[, 13]))


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
library(class)
# using 6-folds cv
wine_id <- createFolds(norm_wine$qual5, 6)
# tuning parameters odd k's only!
kays = seq(1, 33, 2) 

knn.tune <- function(data, x, y, cvid, kays, ...) {
  cv_out <- matrix(NA,nrow=nrow(data),ncol=length(kays)) 
  for (b in 1:length(kays)) {
    for (a in names(cvid)) {
      ids <- cvid[[a]]
      cv_out[ids, b] <- knn(data[-ids, x], data[ids, x], 
                            data[-ids, y], k=kays[b])
    }
  }
  cv_out
}


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
train_knn <- knn.tune(norm_wine, c(1:11), c(12), wine_id, kays)

ercv1 <- apply(train_knn, 2, error.rate, 
               ref=as.integer(norm_wine$qual5))

# best k
(best_k <- kays[which.min(ercv1)])


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
pred_knn <- knn(norm_wine[, -12], norm_test_wine[,-12], norm_wine[,12],
                best_k)
table(norm_test_wine$qual5, pred_knn)
error.rate(pred_knn, norm_test_wine$qual5)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
library(randomForest)
library(doSNOW)
library(latticeExtra)

# Number of randomly sampled variable at each split
# Minimum size of terminal nodes
rf_grid <- expand.grid(.mtry=1:4,           
                    .nodesize=1:10)         
pwine <- table(wine$qual5) / length(wine$qual5)

# It will takes the average class 3 times the vote in order to win
(pinf <- -pwine*log(pwine))


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)
infout<-foreach(a=1:dim(rf_grid)[1],.combine='rbind',
                .packages='randomForest') %dopar% 
{
  tmp = randomForest(qual5~.
                     , data=wine[,-12], ntree=200                    
                     # the cutoff is based on information entrophy
                     , cutoff=pinf/sum(pinf), mtry=rf_grid$.mtry[a]
                     , nodesize=rf_grid$.nodesize[a])
  data.frame(thresh="Information"
             , m=rf_grid$.mtry[a]
             , nodesize=rf_grid$.nodesize[a]
             , err=error.rate(tmp$predicted,wine$qual5))
}

stopCluster(cl)
registerDoSEQ()


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
levelplot(err~nodesize+m
          , data=infout
          , panel=panel.levelplot.points
          , cex=3,pch=22,col.regions=rainbow(31,start=2/3)
          , main="Random-Forest Prediction Error Matrix")+
layer_(panel.2dsmoother(...,col.regions=rainbow(31,start=2/3,alpha=0.7)))


## ------------------------------------------------------------------------
rf_train <- randomForest(qual5~., 
                         data=wine[,-12], 
                         # tree to grow
                         ntree=500,                             
                         # cutoff for winning class
                         cutoff=round(pinf/sum(pinf),3),        
                         # the higher the less detail
                         nodesize=10,               
                         # number of randomly sampled var at each split
                         mtry=1)                    


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
# This plot shows that 1 parameter is good enough for each split since
# the predictive contribution is evenly distributed
varImpPlot(rf_train, pch=10, col=4)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
rf_train$confusion
error.rate(rf_train$predicted, wine$qual5)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
rf_pred <- predict(rf_train, test_wine[,c(-12, -13)])
table(test_wine$qual5, rf_pred)
error.rate(rf_pred, test_wine$qual5)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
library(doMC)
library(parallel)
registerDoMC(detectCores()-2)

# setup 6-fold repeated CV
ctrl <- trainControl(method='cv',
                     number=6)

# to ensure the two models can be compared
set.seed(1234)

knn_tune <- train(wine[,-c(12,13)],
                 wine$qual5,
                 method = "knn",
                 # Center and scaling will occur for new predictions too
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(.k = kays),
                 trControl = ctrl)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
library(ggplot2)
# default metric for classification is 'accuracy'
# dev.off()
ggplot(knn_tune) + theme(legend.position = "top")


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
knn_pred <- predict(knn_tune, newdata = test_wine[,-c(12,13)])

table(test_wine$qual5, knn_pred)
error.rate(knn_pred, test_wine$qual5)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
# the version doesn't support nodesize
rf_tune <- train(wine[,-c(12,13)],
                 wine$qual5,
                 method = "rf",
                 tuneGrid = expand.grid(.mtry=1:4),
                 trControl = ctrl)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
ggplot(rf_tune) + theme(legend.position = "top")


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
plot(varImp(rf_tune, scale = F))


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
rf_pred <- predict(rf_tune, newdata = test_wine[,-c(12,13)])
table(test_wine$qual5, rf_pred)
error.rate(rf_pred, test_wine$qual5)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
resamps <- resamples(list(KNN = knn_tune,
                          RF = rf_tune))
#summary(resamps)
difValues <- diff(resamps)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
summary(difValues)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
lpRF <- getModelInfo(model = "rf", regex = FALSE)[[1]]
lpRF$parameters <- data.frame(parameter = c("mtry", "nodesize"),
                              class = rep("numeric", 2),
                              label = c("mtry", "nodesize"))

lpRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x = as.matrix(x), y = y,
               ntree=500,
               nodesize=param$nodesize,
               mtry=param$mtry,...)
}


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
new_rf_tune <- train.default(wine[,-c(12,13)],
                 wine$qual5,
                 method=lpRF,
                 # optional, otherwise the default is used
                 tuneGrid=rf_grid,    
                 trControl=ctrl)


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
ggplot(new_rf_tune) + theme(legend.position = "top")


## ----fig.width=10, out.width=700, out.height=300, message=FALSE----------
new_rf_pred <- predict(new_rf_tune, newdata = test_wine[,-c(12,13)])
table(test_wine$qual5, new_rf_pred)
error.rate(new_rf_pred, test_wine$qual5)


