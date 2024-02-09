library(AUC)
library(e1071)
library(ROSE)
library(nnet)
library(tree)

# read data into memory
training_digits <- read.csv("training_data.csv", header = TRUE)
training_labels <- read.csv("training_labels.csv", header = FALSE)

# get X and y values
X <- as.matrix(training_digits)
y_truth <- training_labels[,1]

# get number of samples and number of features
N <- length(y_truth)
D <- ncol(X)

# calculate the covariance matrix
Sigma_X <- cov(X)

# calculate the eigenvalues and eigenvectors
decomposition <- eigen(Sigma_X, symmetric = TRUE)

# plot scree graph
plot(1:D, decomposition$values, 
     type = "l", las = 1, lwd = 2,
     xlab = "Eigenvalue index", ylab = "Eigenvalue")

# plot the proportion of variance explained
pove <- cumsum(decomposition$values) / sum(decomposition$values)
plot(1:D, pove, 
     type = "l", las = 1, lwd = 2,
     xlab = "R", ylab = "Proportion of variance explained")
abline(h = 0.95, lwd = 2, lty = 2, col = "blue")
abline(v = which(pove > 0.95)[1], lwd = 2, lty = 2, col = "blue")

# calculate two-dimensional projections
Z <- (X - matrix(colMeans(X), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:9]

firstData <- cbind(Z,training_labels)
colnames(firstData) <- c("A1", "A2", "A3", "A4", "A5", "A6","A7", "A8", "A9", "V1")

#library(ROSE)
# balanced data set with both over and under sampling
data.balanced.under <- ovun.sample(V1~., data= firstData, 
                                   method="under", N = 90700, seed = 1)$data

table(data.balanced.under$V1)
N <- length(data.balanced.under$V1)

#5x2 cross validation parameters
data <- data.balanced.under
data <- data[sample(nrow(data)),]
disc_data <- split(data,data$V1) 
notopen <- disc_data[[1]]
open <- disc_data[[2]]
numberopen <- sum(open[,10])
numbernotopen <- N - numberopen

#scaling
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
scaled <- 2*(scaled) -1 

disc_scaled <- split(scaled,scaled$V1) 
notopen <- disc_scaled[[1]]
open <- disc_scaled[[2]]

#g1 for training set
g1  <- array(0, dim = c(45350,10,6))
#g2 for validation set
g2  <- array(0, dim = c(45350,10,6))

# first data group for validation
for(i in seq(1, 5, 2)){

G1 <- rbind(open[1:(numberopen/2),1:10],notopen[1:(numbernotopen/2),1:10])
G2 <- rbind(open[(1+numberopen/2):(2*numberopen/2),1:10],notopen[(1+numbernotopen/2):(2*numbernotopen/2),1:10])

G1 <- G1[sample(nrow(G1)),]
G2 <- G2[sample(nrow(G2)),]


g1[,,i]  <- as.matrix(G1)
g2[,,i]  <- as.matrix(G2)

g1[,,i+1]  <- as.matrix(G1)
g2[,,i+1]  <- as.matrix(G2)

scaled <- scaled[sample(nrow(scaled)),]
disc_scaled <- split(scaled,scaled$V1) 
notopen <- disc_scaled[[1]]
open <- disc_scaled[[2]]

}

NBresults <- matrix(rep(0,5), nrow = 5)
MLPresults <- matrix(rep(0,15), nrow = 3, ncol = 5)
DTresults <- matrix(rep(0,5), nrow = 5)
ThreeResults <- matrix(rep(0,5), nrow = 5)
TwoResults <- matrix(rep(0,5), nrow = 5)


#naivebayes as a benchmark
for(d in 1:5) {

#library(e1071) 
A <- g1[,,d]
B <- g2[,,d]
A <- as.data.frame(A)
B <- as.data.frame(B)
model <- naiveBayes(V10 ~ ., data = A)
default_pred <- predict(model, B, type="raw")
xxx <- as.factor(B$V10)
#library(AUC)
roc_curve_nb <- roc(predictions = default_pred[,2], labels = xxx)
auc(roc_curve_nb)

NBresults[d] <- auc(roc_curve_nb)



# MLP analysis
#library(nnet)
A <- g1[,,d]
B <- g2[,,d]
A <- as.data.frame(A)
B <- as.data.frame(B)


aa1 <- A[,-10]
a1 <- as.matrix(aa1)
aa2 <- B[,-10]
a2 <- as.matrix(aa2)

l1 <- A[,10]
x1 <- class.ind(l1)
l2 <- B[,10]
l2 <- factor(l2)

# cross validation for multilayer perceptron
sizeVector <- c(1,20,40)

for(j in 1:3){

modelMLP <- nnet(a1, x1, size= sizeVector[j])
predictionsMLP <- predict(modelMLP, a2, type="raw")

roc_curve_MLP <- roc(predictions = predictionsMLP[,2], labels = l2)
auc(roc_curve_MLP)

MLPresults[j,d] <- auc(roc_curve_MLP)
}


# tree analysis

#library(tree)

ll1 <- factor(l1)

tree_classifier <- tree(V10 ~ ., data = cbind(aa1, V10 = ll1))
training_scores <- predict(tree_classifier, aa2)
roc_curve_tree <- roc(predictions = training_scores[,2], labels = xxx)
auc(roc_curve_tree)

DTresults[d] <- auc(roc_curve_tree)


# combining learners
  # all of them 

CombineTrain <- (training_scores + predictionsMLP + default_pred) / 3
roc_curve_three <- roc(predictions = CombineTrain[,2], labels = xxx)
auc(roc_curve_three)

ThreeResults[d] <- auc(roc_curve_three)


  # Naive Bayes and MLP
CombineTrain <- (predictionsMLP + default_pred) / 2
roc_curve_two <- roc(predictions = CombineTrain[,2], labels =xxx)
auc(roc_curve_two)

TwoResults[d] <- auc(roc_curve_two)

}


NBmean <- colMeans(NBresults)
MLPmean <- colMeans(t(MLPresults))
DTmean <- colMeans(DTresults)
Threemean <- colMeans(ThreeResults)
Twomean <- colMeans(TwoResults)

MLPmean <- MLPmean[which.max(MLPmean)]

#Result size for MLP = 20

means <- c(NBmean, MLPmean, DTmean, Threemean, Twomean)
which.max(means)

#optimal solution is MLP with size = 20

#IMPLEMENTATION OF THE OPTIMUM METHOD ON TESTING SAMPLE
X_test <- read.csv("test_data.csv", header = TRUE)
X <- as.matrix(X_test)

# get number of samples and number of features
N <- length(X_test$F01)
D <- ncol(X)
# calculate the covariance matrix
Sigma_X <- cov(X)

# calculate the eigenvalues and eigenvectors
decomposition <- eigen(Sigma_X, symmetric = TRUE)
# calculate two-dimensional projections
X_test_dim_red <- (X - matrix(colMeans(X), N, D, byrow = TRUE)) %*% decomposition$vectors[,1:9]

#scaling for test
maxs <- apply(X_test_dim_red, 2, max) 
mins <- apply(X_test_dim_red, 2, min)
scaled_test <- as.data.frame(scale(X_test_dim_red, center = mins, scale = maxs - mins))
scaled_test <- 2*(scaled_test) -1 

#MLP
scaled_test <- as.matrix(scaled_test)

aa1 <- scaled[,-10]
a1 <- as.matrix(aa1)

l1 <- scaled[,10]
x1 <- class.ind(l1)

test_classifier <- nnet(a1, x1, size= 20)
test_scores <- predict(test_classifier, scaled_test, type="raw")
write.table(test_scores[,2], file = "test_predictions.csv", row.names = FALSE, col.names = FALSE)