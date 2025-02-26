


# Load data
setwd("C:/J/fastfood")
library(bootstrap)
library(jsonlite)
filename = "testing_performance/Wednesday_plots/extracted_data.csv"
data <- readLines(filename)  # Read file as text

# Find header lines
header_lines <- grep("^#", data)
matrices <- list()
names <- list()

# Process each matrix
for (i in seq_along(header_lines)) {
  start <- header_lines[i] + 1  # Start after header
  end <- if (i < length(header_lines)) header_lines[i + 1] - 1 else length(data)  # End before next header or EOF
  
  # Read the matrix data
  matrix_data <- read.csv(text = paste(data[start:end], collapse = "\n"), header = FALSE)
  matrices[[i]] <- as.matrix(matrix_data)
  
  # Read the matrix name
  matrix_name <- read.csv(text = paste(data[header_lines[i]], collapse = "\n"), header = FALSE)
  names[[i]] <- matrix_name
}
names <- sapply(names, function(x) x$V1)
storage = list()

# Iterate matrices
for (i in 1:length(matrices)) {
  matrix <- matrices[[i]]
  trial.CI = c()
  
  # Iterate trials
  for (j in 1:nrow(matrix)) {
    vect <- matrix[j,]
    CI <- non.parametric.bootstrap(vect)
    sample.mean <- mean(vect)
    inner.data = c(sample.mean, as.numeric(CI))
    trial.CI = c(trial.CI, inner.data)
  }
  trial.CI <- matrix(trial.CI, nrow=3)
  filename <- paste0(gsub("^# |\\.pkl$", "", names[[i]]), ".csv")
  print(filename)
  write.csv(trial.CI, file = filename, row.names=FALSE)
}

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
  if (any(variance.estimates == 0)) {
    warning("Variance estimates contain zeros, which may cause NaN in tb.")
  }
  
  # Find lower & upper percentile estimates for t*
  tb = (theta.b - grand.mean) / sqrt(variance.estimates)
  t.star = quantile(tb, probs=c(1-alpha/2, alpha/2), na.rm=TRUE)
  
  percentiles = grand.mean - t.star * sqrt(var.theta)
  c(percentiles[1], percentiles[2])
}

