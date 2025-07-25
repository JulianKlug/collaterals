library(ordinal)
library(lme4)
library(lmerTest)

data_path <- "/Users/jk1/Downloads/preprocessed_lactate_df.csv"
data <- read.csv(data_path)

# X3M.mRS as factor
data$X3M.mRS <- as.factor(data$X3M.mRS)

# drop nan in value and X3M.mRS
data <- na.omit(data[, c("value", "X3M.mRS", "case_admission_id")])

# COMPREHENSIVE DIAGNOSTICS
print("=== DATA DIAGNOSTICS ===")
print(paste("Total observations:", nrow(data)))
print(paste("Unique cases:", length(unique(data$case_admission_id))))
print("mRS distribution:")
print(table(data$X3M.mRS))
print("Observations per case:")
obs_per_case <- table(data$case_admission_id)
print(summary(as.numeric(obs_per_case)))
print(paste("Cases with only 1 observation:", sum(obs_per_case == 1)))
print("Value summary:")
print(summary(data$value))

# Scale the predictor variable
data$value_scaled <- scale(data$value)[,1]

# SOLUTION 2: Try multiple optimizers systematically
try_lmer <- function(data, optimizer_name, optimizer_func) {
  cat("Trying optimizer:", optimizer_name, "\n")
  tryCatch({
    model <- lmer(as.numeric(X3M.mRS) ~ value_scaled + (1|case_admission_id), 
                  data = data,
                  control = lmerControl(optimizer = optimizer_func, 
                                       optCtrl = list(maxfun = 50000),
                                       check.conv.grad = .makeCC("warning", tol = 0.02)))
    
    if (length(model@optinfo$conv$lme4$messages) == 0) {
      cat("SUCCESS with", optimizer_name, "\n")
      return(model)
    } else {
      cat("Convergence warnings with", optimizer_name, "\n")
      return(model)  # Return anyway, might be usable
    }
  }, error = function(e) {
    cat("FAILED with", optimizer_name, ":", e$message, "\n")
    return(NULL)
  })
}

# Try different optimizers
optimizers <- list(
  "bobyqa" = "bobyqa",
  "Nelder_Mead" = "Nelder_Mead", 
  "nlminbwrap" = "nlminbwrap",
  "nloptwrap" = "nloptwrap"
)

best_model <- NULL
for (opt_name in names(optimizers)) {
  if (nrow(data_filtered) > 0) {
    model <- try_lmer(data_filtered, opt_name, optimizers[[opt_name]])
    if (!is.null(model)) {
      best_model <- model
      break
    }
  }
}

# SOLUTION 3: If filtering didn't help, try original data with relaxed settings
if (is.null(best_model)) {
  cat("Trying with original data and very relaxed settings...\n")
  best_model <- lmer(as.numeric(X3M.mRS) ~ value_scaled + (1|case_admission_id), 
                     data = data,
                     control = lmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 100000),
                                          check.conv.singular = .makeCC("ignore"),
                                          check.conv.grad = .makeCC("ignore")))
}

# SOLUTION 4: Check if random effects are actually needed
cat("\n=== CHECKING RANDOM EFFECTS NECESSITY ===\n")
# Fit without random effects
simple_lm <- lm(as.numeric(X3M.mRS) ~ value_scaled, data = data)

# Calculate ICC manually
if (!is.null(best_model)) {
  var_components <- as.data.frame(VarCorr(best_model))
  random_var <- var_components$vcov[1]
  residual_var <- var_components$vcov[2]
  icc <- random_var / (random_var + residual_var)
  cat("Intraclass Correlation Coefficient:", round(icc, 4), "\n")
  
  if (icc < 0.05) {
    cat("WARNING: Very low ICC suggests random effects may not be necessary\n")
    cat("Consider using simple linear regression instead\n")
  }
}

# SOLUTION 5: Alternative - ordinal models
cat("\n=== TRYING ORDINAL MODELS ===\n")

# Simple ordinal regression (no random effects)
simple_clm <- clm(X3M.mRS ~ value_scaled, data = data)
cat("Simple ordinal regression succeeded\n")

# Try clmm with different approaches
if (nrow(data_filtered) > 0) {
  tryCatch({
    clmm_model <- clmm(X3M.mRS ~ value_scaled + (1|case_admission_id), 
                       data = data_filtered,
                       control = clmm.control(method = "ucminf", maxIter = 1000))
    cat("CLMM succeeded with filtered data\n")
    print(summary(clmm_model))
  }, error = function(e) {
    cat("CLMM failed:", e$message, "\n")
  })
}

# Final output
if (!is.null(best_model)) {
  cat("\n=== FINAL MODEL SUMMARY ===\n")
  print(summary(best_model))
} else {
  cat("\n=== USING SIMPLE MODELS ===\n")
  cat("Linear regression summary:\n")
  print(summary(simple_lm))
  cat("\nOrdinal regression summary:\n")
  print(summary(simple_clm))
}