


# Load data
setwd("C:/J/fastfood")
library(bootstrap)
library(jsonlite)
filename = "testing_performance/batch_norm_False.csv"
data = read.csv(filename, header = T, stringsAsFactors = T)
# Convert factor to character, then parse each row as a numeric vector
set1 <- lapply(as.character(data$train_accuracy), fromJSON)
set2 <- lapply(as.character(data$test_accuracy), fromJSON)

sets <- list(train = set1, test = set2)
for (set_name in names(sets)) {
  set <- sets[[set_name]]
  
  confidence.intervals <- lapply(set1, function(x) {
    # call Monte Carlo resampling.
    if (length(x) == 1) {
      return(c(0,0)) # Handle case when data is a single element
    }
    non.parametric.bootstrap(x)
  })
  # Unpack confidence.intervals (good for debugging b/c it is readable)
  # confidence.intervals <- unlist(confidence.intervals)
  # confidence.intervals <- matrix(confidence.intervals, ncol=2, byrow=T)
  
  # Store confidence.interavals into the dataframe as a new column
  # Use JSON formatting so it gets stored as a list[float]
  data[[paste0(set_name, ".CI")]] <- sapply(confidence.intervals, toJSON)
}
write.csv(data, "testing_performance/batch_norm_False_2.csv", row.names=FALSE)

no.resampling <- function(population, alpha=0.05) {
  # No Resampling Confidence Interval
  n = length(population)
  mean = mean(population)
  variance = mean((population - mean)^2)
  t.stat = qt(1-alpha/2, df=n-1)
  CI.lower = mean - t.stat * sqrt(variance / n)
  CI.upper = mean + t.stat * sqrt(variance / n)
  c(CI.lower, CI.upper)
}

non.parametric.bootstrap <- function(population, B=10000, alpha=0.05) {
  # Non-Parametric Percentile Bootstrap
  n = length(population)
  resample = sample(population, size = B*n, replace = TRUE)
  resample = matrix(resample, nrow = B)
  mc.mean = rowMeans(resample)
  grand.mean = mean(mc.mean)
  mc.variance = mean((mc.mean - grand.mean)^2)
  percentiles = quantile(mc.mean, probs = c(alpha/2, 1 - alpha/2))
  c(percentiles[1], percentiles[2])
}

semi.parametric.bootstrap <- function(population, B=10000, alpha=0.05) {
  # Semi-Parametric Percentile Bootstrap
  # The inverse of the sample mean is an estimate for theta. Generate Exp(1/x.bar)
  n = length(population)
  x.bar = mean(population)
  resample = rexp(B*n, rate = 1/x.bar)
  resample = matrix(resample, nrow=B)
  mc.mean = rowMeans(resample)
  grand.mean = mean(mc.mean)
  mc.variance = mean((mc.mean - grand.mean)^2)
  percentiles = quantile(mc.mean, probs = c(alpha/2, 1 - alpha/2))
  c(percentiles[1], percentiles[2])
}

studentized.bootstrap <- function(population, B=1000, C=200, alpha=0.05) {
  # Studentized Bootstrap
  n = length(population)
  grand.mean = mean(population)
  resample = sample(population, size = B*n, replace=TRUE)
  resample = matrix(resample, nrow=B)
  theta.b = rowMeans(resample)
  var.theta = mean((theta.b - grand.mean)^2)
  
  # Compute estimates for variance(theta.b)
  variance.estimates <- apply(resample, 1, function(row) {
    inner.resample = sample(row, size = C*n, replace=TRUE)
    inner.resample = matrix(inner.resample, nrow=C)
    theta.bc = rowMeans(inner.resample)
    theta.b.bar = mean(theta.bc)
    mean((theta.bc - theta.b.bar)^2) # Variance for each b block
  })
  # if (any(variance.estimates == 0)) {
  #   warning("Variance estimates contain zeros, which may cause NaN in tb.")
  # }
  
  # Find lower & upper percentile estimates for t*
  tb = (theta.b - grand.mean) / sqrt(variance.estimates)
  t.star = quantile(tb, probs=c(1-alpha/2, alpha/2), na.rm=TRUE)
  
  percentiles = grand.mean - t.star * sqrt(var.theta)
  c(percentiles[1], percentiles[2])
}

