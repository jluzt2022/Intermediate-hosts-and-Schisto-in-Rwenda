
#Rcodes for analysis of the data from Rwanda
#Schistosomiasis transmission: 
#A machine learning analysis reveals the importance of Agrochemicals on snail abundance in Rwanda

# -------------------- Load Necessary Libraries -------------------- #
# Load necessary libraries

# Machine Learning Libraries
library(randomForest)     # For Random Forest
library(ranger)           # For fitting Random Forests
library(rfUtilities)      # For utility functions related to Random Forest
library(rfPermute)        # For Random Forest with permutation tests
library(xgboost)          # For fitting Gradient Boosting Machines (GBMs)
library(rpart)            # For fitting CART-like decision trees
library(class)            # For K-Nearest Neighbors (KNN)
library(earth)            # For Multivariate Adaptive Regression Splines (MARS)
library(gbm)              # For Gradient Boosting Machines

# Model Evaluation and Performance Libraries
library(caret)            # For cross-validation and model tuning
library(pROC)             # For ROC and AUC calculation
library(ROCR)             # For calculating TSS and other metrics
library(ROSE)             # For data balancing, used in MARS models

# Statistical Modeling Libraries
library(lme4)             # For Generalized Linear Mixed Models (GLMM)
library(car)              # For regression diagnostics

# Utility Libraries
library(reshape2)         # For data reshaping
library(pdp)              # For partial dependence plots
library(vip)              # For variable importance plots
library(Ckmeans.1d.dp)    # For optimal k-means clustering

# Plotting Libraries
library(ggplot2)          # For creating plots
library(patchwork)        # For combining plots
library(gridExtra)        # For arranging multiple plots
library(viridis)          # For color scales
library(sjPlot)           # For visualizing regression models
library(ggeffects)        # For computing marginal effects and predictions
library(plotmo)           # For MARS model plotting
library(plotrix)          # For various plot functions

#Upload the excell.txt file (dataset)

dataset <- read.table("C:/Users/jlu-su/Desktop/Kalinda_folder/dataset.txt", header = TRUE, sep = "\t")

# define response variables as factors (if a factor, classification is assumed, otherwise regression (we want classification))
dataset$Bulinus <- as.factor(dataset$Bulinus)
dataset$Biom <- as.factor(dataset$Biom)
dataset$BulBio <- as.factor(dataset$BulBio)
#Covert particular columns  predictors as factors
dataset$Water_body<- as.factor(dataset$Water_body)
dataset$Elevetion <- as.factor(dataset$Elevetion)

#Preliminary analysis
#Check for missing values
#Check for duplicate rows 
#collinearity in the dataset
{
  dataset <- read.table("C:/Users/jlu-su/Desktop/Kalinda_folder/dataset.txt", header = TRUE, sep = "\t")
  # Convert necessary variables to factors
  #dataset$Elevetion <- as.factor(dataset$Elevetion)
  dataset$Water_body <- as.factor(dataset$Water_body)
  dataset$Bulinus <- as.factor(dataset$Bulinus)
  dataset$Biom <- as.factor(dataset$Biom)
  
  # Check for missing values and remove rows with missing data
  cat("Checking for missing values...\n")
  if (sum(is.na(dataset)) > 0) {
    cat("Removing rows with missing values...\n")
    dataset <- na.omit(dataset)
  } else {
    cat("No missing values found.\n")
  }
  
  # Ensure all predictors are numeric before scaling
  pred <- c("PH", "T0C", "E.C.µs.cm", "TDS.mg.l", "D.O.mg.l", "SO4", "Cl", 
            "NO2", "NO3", "NH4", "PO4", "TN", "TP", "TH", "CaH", "Ca", "Mg", 
            "Na", "K", "Pb", "Cd", "Mn", "Fe", "Zn", "Elevetion")
  # Check for multicollinearity using correlation matrix
  cat("Checking for multicollinearity...\n")
  cor_matrix <- cor(dataset[pred])
  corrplot(cor_matrix, method = "color", type = "upper", 
           tl.cex = 0.8, addCoef.col = "black")
  
  # Open a graphics device if needed (like when running in script mode)
  dev.new()  # Opens a new plotting window, use it only if needed
  
  # Plot correlation matrix
  corrplot(cor_matrix, method = "color", type = "upper", 
           tl.cex = 0.8, addCoef.col = "black",number.cex = 0.7)
  # Compute the correlation matrix
  cor_matrix <- cor(dataset[pred])
  library(caret)
  
  # Find highly correlated predictors
  high_corr_indices <- findCorrelation(cor_matrix, cutoff = 0.9)  # Adjust the cutoff as needed
  high_corr_vars <- colnames(cor_matrix)[high_corr_indices]
  
  cat("Highly correlated predictors:\n")
  print(high_corr_vars)
  
  ####
  #Find the Most Correlated Pairs
  # Convert the correlation matrix to a data frame
  cor_melted <- as.data.frame(as.table(cor_matrix))
  # Remove self-correlations (i.e., correlation of a variable with itself)
  cor_melted <- cor_melted[cor_melted$Var1 != cor_melted$Var2, ]
  # Sort by absolute correlation value
  cor_melted <- cor_melted[order(-abs(cor_melted$Freq)), ]
  # Display the top correlated pairs
  head(cor_melted)
  #
  #Visualize the Correlations
  library(corrplot)
  # Plot the correlation matrix with numbers
  corrplot(cor_matrix, method = "color", type = "upper", 
           tl.cex = 0.8,         # Size of text labels for predictors
           number.cex = 0.5,     # Size of numbers in the plot
           addCoef.col = "black") # Color of the coefficients
  dev.off()
  
  # Save to PDF
  pdf("correlation_plot.pdf")
  corrplot(cor_matrix, method = "color", type = "upper", 
           tl.cex = 0.8, addCoef.col = "black" ,number.cex = 0.5)
  dev.off()
  
  # Save to PNG
  png("correlation_plot.png", width = 800, height = 600)
  corrplot(cor_matrix, method = "color", type = "upper", 
           tl.cex = 0.8, addCoef.col = "black")
  dev.off()
  
  ## Example: Removing one of the highly correlated variables
  dataset <- dataset[, !(names(dataset) %in% c("CaH"))]
  dataset <- dataset[, !(names(dataset) %in% c("TDS.mg.l"))]
}


#######################################################################
#...............Evaluate the performance of 5 different models and choose the good ones................
#...........We compare Random Forest, Decision Tree, KNN, GBM (e.g XGBoost) and MARS models.............
#................We do this by calculating the AUC and TSS values............................. 
#First of all remember we are working with the clean dataset
#Where we just alternate response_vars
# Before Model fitting
# Ensure all predictors are numeric before scaling
predictors <- c("PH", "T0C", "E.C.µs.cm", "D.O.mg.l", "SO4", "Cl", 
                "NO2", "NO3", "NH4", "PO4", "TN", "TP", "TH", "Ca", "Mg", 
                "Na", "K", "Pb", "Cd", "Mn", "Fe", "Zn")

set.seed(123)
# Splitting data into training and testing (70-30 split)
library(caret)
trainIndex <- createDataPartition(dataset$Biom, p = .7, list = FALSE, times = 1)
trainData <- dataset[trainIndex, ]
testData  <- dataset[-trainIndex, ]

# Rename the column in trainData and replace due to spelling issue
names(trainData)[names(trainData) == "Elevetion"] <- "Elevation"
# Rename the column in testData
names(testData)[names(testData) == "Elevetion"] <- "Elevation"
#Change the dataset name too
names(dataset)[names(dataset) == "Elevetion"] <- "Elevation"
# Assuming 'factor_vars' is a vector of names of categorical variables
factor_vars <- c("Elevation") # Update with actual categorical variables
# Convert categorical variables in train and test datasets to factors with consistent levels
for (var in factor_vars) {
  trainData[[var]] <- factor(trainData[[var]])
  testData[[var]] <- factor(testData[[var]], levels = levels(trainData[[var]]))
}

# Example: scaling predictors in train and test data
trainData_scaled <- as.data.frame(scale(trainData[, predictors]))
testData_scaled <- as.data.frame(scale(testData[, predictors]))

# Adding the target variable back
trainData_scaled$Biom <- trainData$Biom
testData_scaled$Biom <- testData$Biom

# Convert categorical variables to factors if needed
trainData$Elevation <- as.factor(trainData$Elevation)
testData$Elevation <- as.factor(testData$Elevation)

# Ensure correct response and predictor variables
response <- "Biom"
predictors <- colnames(dataset)[!colnames(dataset) %in% c("Biom", "Elevation")]

# Imputation for missing values in Elevation (numeric conversion)
trainData$Elevation <- as.numeric(as.character(trainData$Elevation))
testData$Elevation <- as.numeric(as.character(testData$Elevation))

# Mean imputation for Elevation
testData$Elevation[is.na(testData$Elevation)] <- mean(trainData$Elevation, na.rm = TRUE)

# --- Model Fitting ---  

# Random Forest Model
rf_model <- randomForest(Biom ~ ., data = trainData[, c(predictors, response)], ntree = 500)

# Decision Tree Model
model_tree <- rpart(Biom ~ ., data = trainData[, c(predictors, response)], method = "class")

# KNN Model
k <- 3
knn_predictions <- knn(train = trainData[, predictors], 
                       test = testData[, predictors], 
                       cl = trainData[, response], k = k)
# KNN Model
dtrain <- xgb.DMatrix(data = as.matrix(trainData[, predictors]), label = as.numeric(trainData$Biom) - 1)
final_model <- xgboost(data = dtrain, objective = "binary:logistic", nrounds = 50, max_depth = 3, eta = 0.1)

# MARS Model
balanced_data <- ovun.sample(Biom ~ ., data = trainData, method = "over", N = 58)$data
mars_model <- earth(Biom ~ ., data = balanced_data[, c(predictors, response)], degree = 1, glm = list(family = binomial, maxit = 100))

# --- Function to Calculate AUC ---
calculate_auc <- function(model, data, response, predictors, model_type) {
  if (model_type == "rf") {
    # Random Forest predictions
    predictions <- predict(model, newdata = data, type = "prob")[, 2]
  } else if (model_type == "tree") {
    # Decision Tree predictions
    predictions <- predict(model, newdata = data, type = "prob")[, 2]
  } else if (model_type == "knn") {
    # KNN predictions are already passed in as a factor (model is predictions)
    predictions <- as.numeric(model) - 1  # Convert factor to numeric (0/1)
  } else if (model_type == "xgboost") {
    # XGBoost model predictions
    predictions <- predict(model, as.matrix(data[, predictors]))
  } else if (model_type == "mars") {
    # MARS model predictions
    predictions <- predict(model, newdata = data, type = "response")
  } else {
    stop("Unknown model type")
  }
  
  # Ensure predictions are of the correct length
  if (length(predictions) != nrow(data)) {
    stop(paste("Length of predictions does not match the number of rows in the test data for model type:", model_type))
  }
  
  roc_curve <- roc(data[[response]], predictions)
  auc <- auc(roc_curve)
  return(auc)
}

# --- Function to Calculate TSS ---
calculate_tss <- function(predictions, response_values) {
  # Convert predictions to binary class using threshold 0.5
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)
  
  # Ensure predicted_classes length matches response length
  if (length(predicted_classes) != length(response_values)) {
    stop("Length of predicted classes does not match response length")
  }
  
  # Calculate confusion matrix
  confusion <- table(factor(predicted_classes, levels = c(0, 1)), factor(response_values, levels = c(0, 1)))
  
  # Handle cases where confusion matrix entries might be zero
  TP <- ifelse(!is.na(confusion["1", "1"]), confusion["1", "1"], 0)
  FP <- ifelse(!is.na(confusion["1", "0"]), confusion["1", "0"], 0)
  TN <- ifelse(!is.na(confusion["0", "0"]), confusion["0", "0"], 0)
  FN <- ifelse(!is.na(confusion["0", "1"]), confusion["0", "1"], 0)
  
  # Calculate TSS
  TSS <- (TP / (TP + FN)) - (FP / (FP + TN))
  return(TSS)
}

# --- Function to Generate Predictions for TSS ---
generate_predictions <- function(model, data, response, predictors, model_type) {
  if (model_type == "rf") {
    return(predict(model, newdata = data, type = "prob")[, 2])
  } else if (model_type == "tree") {
    return(predict(model, newdata = data, type = "prob")[, 2])
  } else if (model_type == "knn") {
    return(as.numeric(model) - 1)
  } else if (model_type == "xgboost") {
    return(predict(model, as.matrix(data[, predictors])))
  } else if (model_type == "mars") {
    return(predict(model, newdata = data, type = "response"))
  } else {
    stop("Unknown model type")
  }
}
# --- Loop over Models to Calculate AUC and TSS ---
models <- list(rf_model, model_tree, knn_predictions, final_model, mars_model)
model_types <- c("rf", "tree", "knn", "xgboost", "mars")
model_names <- c("Random Forest", "Decision Tree", "KNN", "XGBoost", "MARS")

auc_results <- numeric(length(models))
tss_results <- numeric(length(models))

for (i in 1:length(models)) {
  model_type <- model_types[i]
  
  # Generate predictions
  predictions <- generate_predictions(models[[i]], testData, response, predictors, model_type)
  
  # Calculate AUC
  auc_results[i] <- calculate_auc(models[[i]], testData, response, predictors, model_type)
  
  # Calculate TSS
  tss_results[i] <- calculate_tss(predictions, testData[[response]])
}

# --- Show the Results ---
comparison <- data.frame(Model = model_names, AUC = auc_results, TSS = tss_results)
print(comparison)




##################################################################################
#CROSS VALIDATION, COMPARISION OF RANDOM FOREST AND XGBOOST MODELS
#Combiing both RF and Xgboost because the are high performing ones 
#Using the clean and intermediate dataset for final analysis
#VARIABLE IMPORTANCE
# Function to run cross-validation models and plot variable importance for any response variable
run_cv_and_plot <- function(response_var) {
  # Define the target and predictors
  target <- response_var
  predictors <- setdiff(names(dataset), target)
  
  # Remove unnecessary variables
  dataset <- dataset[, !(names(dataset) %in% c("CaH", "TDS.mg.l", "Water_body"))]
  # Switch to different response_var, ie to identify VI for Bulinus and first remove response_vars; Biom and BulBio
  #Repeat the process for the Biom and BulBio 
  dataset <- dataset[, !(names(dataset) %in% c( "BulBio", "Biom"))]
  
  
  # -------------------- Cross-Validation Setup -------------------- #
  set.seed(102)
  
  # Set up cross-validation
  cv_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
  
  # -------------------- XGBoost Model with Cross-Validation -------------------- #
  xgb_grid <- expand.grid(
    nrounds = 100,
    max_depth = 5,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
  
  # Train XGBoost with cross-validation
  xgb_model <- train(
    as.formula(paste(target, "~ .")), 
    data = dataset, 
    method = "xgbTree", 
    trControl = cv_control,
    tuneGrid = xgb_grid,
    metric = "Accuracy"
  )
  
  # Extract variable importance for XGBoost
  xgb_importance <- varImp(xgb_model)$importance
  if (nrow(xgb_importance) == 0) {
    stop("XGBoost variable importance is empty!")
  }
  xgb_importance <- xgb_importance[order(xgb_importance$Overall), , drop = FALSE]  # Sort in increasing order
  # Rename the unwanted predictor
  rownames(xgb_importance) <- gsub("E.C.Âμs.cm", "E.C.Âµs.cm", rownames(xgb_importance))
  
  # Clean the xgb_importance data frame to remove unwanted predictors
  unwanted_predictor <- "E.C.μs.cm"  # The unwanted predictor name
  xgb_importance_clean <- xgb_importance[!grepl(unwanted_predictor, rownames(xgb_importance)), , drop = FALSE]
  
  # Plot XGBoost variable importance
  xgb_plot <- ggplot(data.frame(Feature = rownames(xgb_importance_clean), Importance = xgb_importance_clean$Overall), 
                     aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = paste("XGBoost Variable Importance for", target), x = "Features", y = "Importance") +
    theme_minimal()
  
  # -------------------- Random Forest Model with Cross-Validation -------------------- #
  rf_model <- train(
    as.formula(paste(target, "~ .")), 
    data = dataset, 
    method = "rf", 
    trControl = cv_control,
    ntree = 100,
    metric = "Accuracy"
  )
  
  # Extract variable importance for Random Forest
  rf_importance <- varImp(rf_model)$importance
  if (nrow(rf_importance) == 0) {
    stop("Random Forest variable importance is empty!")
  }
  rf_importance <- rf_importance[order(rf_importance$Overall), , drop = FALSE]  # Sort in increasing order
  
  # Plot Random Forest variable importance
  rf_plot <- ggplot(data.frame(Feature = rownames(rf_importance), Importance = rf_importance$Overall), 
                    aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "tomato") +
    coord_flip() +
    labs(title = paste("Random Forest Variable Importance for", target), x = "Features", y = "Importance") +
    theme_minimal()
  
  # -------------------- Combine and Return the Plots -------------------- #
  combined_plot <- xgb_plot + rf_plot + plot_layout(ncol = 2)
  return(combined_plot)
}

# -------------------- Choose Response Variable and Run -------------------- #
response_var <- "Bulinus"  # Switch between "Bulinus", "Biom", "BulBio" as needed

# Generate the combined plots for the selected response variable
combined_plot <- run_cv_and_plot(response_var)

# Save the combined plot as PDF
ggsave(paste0("variable_importance_plots_with_cv_", response_var, ".pdf"), combined_plot, width = 10, height = 7)

# Inform user
cat("PDF with cross-validated variable importance plots created and saved.\n")

#PARTIAL ANALYSIS AT WETLAND AND LAKE SHORE SCALE
#Examine the dataset and choose dofferent scale. 
#Choose the scale (wetland =1, lakeshores = 2)
#Choose the response variable and remove the others. 
dataset<- dataset[dataset$Water_body ==2,]# switch between scale, wetlands =1 and lake shores =2
#Rerun the above code to obtain VI at wetland and lakes 

#########################################################
#............response plots..........................................
#...........PDP FOR BOTH RF AND XGBOOST ON THE SAME AXIS.............
# Load necessary libraries
library(pdp)
library(randomForest)
library(xgboost)
library(ggplot2)
library(caret)
library(patchwork)  # For combining plots

# Set the path to the dataset
dataset_path <- "C:/Users/jlu-su/Desktop/Kalinda_folder/dataset.txt"

# Load and preprocess dataset
dataset <- read.table(dataset_path, header = TRUE, sep = "\t")

# Ensure factors are set correctly
dataset$Water_body <- as.factor(dataset$Water_body)
#dataset$Elevetion <- as.factor(dataset$Elevetion)
dataset$Bulinus <- as.factor(dataset$Bulinus)
dataset$Biom <- as.factor(dataset$Biom)
dataset$BulBio <- as.factor(dataset$BulBio)

# Select the response variable
response_variable <- "Biom"  # Change this to "Biom", "BulBio", or any other response as needed

# Remove unnecessary variables (if any)
dataset <- dataset[, !(names(dataset) %in% c("CaH", "TDS.mg.l", "Water_body", "BulBio", "Bulinus"))]

# Define the predictors and the target
predictors <- setdiff(names(dataset), response_variable)
target <- response_variable

# -------------------- Cross-Validation Setup -------------------- #
set.seed(102)

# Set up cross-validation control
cv_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

# -------------------- Random Forest Model -------------------- #
rf_model <- train(
  as.formula(paste(target, "~ .")),
  data = dataset,
  method = "rf",
  trControl = cv_control,
  ntree = 100,
  metric = "Accuracy"
)

# -------------------- XGBoost Model -------------------- #
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 5,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

xgb_model <- train(
  as.formula(paste(target, "~ .")),
  data = dataset,
  method = "xgbTree",
  trControl = cv_control,
  tuneGrid = xgb_grid,
  metric = "Accuracy"
)

# -------------------- PDP Plotting -------------------- #
# Define selected predictors for PDP (show predictors choosen to be important for Biom. Bulinus and BulBiom)
selected_predictors <- c("NH4", "TN", "TH", "E.C.µs.cm")  # Replace with actual predictors of interest

# Create PDPs for both Random Forest and XGBoost

# Random Forest PDP
pdp_rf_list <- lapply(selected_predictors, function(pred) {
  pd_rf <- partial(rf_model, pred.var = pred, plot = FALSE)
  pd_rf$model <- "Random Forest"
  pd_rf
})

# XGBoost PDP
pdp_xgb_list <- lapply(selected_predictors, function(pred) {
  pd_xgb <- partial(xgb_model, pred.var = pred, plot = FALSE)
  pd_xgb$model <- "XGBoost"
  pd_xgb
})

# Combine PDPs for both models for each predictor
pdp_combined <- Map(function(rf, xgb) rbind(rf, xgb), pdp_rf_list, pdp_xgb_list)

# -------------------- Plotting the PDPs -------------------- #
# Generate PDP plots for each selected predictor on the same axis
pdp_plots <- lapply(seq_along(pdp_combined), function(i) {
  predictor <- selected_predictors[i]
  ggplot(pdp_combined[[i]], aes_string(x = predictor, y = "yhat", color = "model")) +
    geom_line(size = 1) +
    labs(title = paste("PDP for", predictor),
         x = predictor, 
         y = "Partial Dependence") +
    theme_minimal() +
    theme(legend.position = "top") +
    scale_color_manual(values = c("Random Forest" = "tomato", "XGBoost" = "steelblue"))
})

# Combine the PDP plots into one figure using patchwork with a horizontal layout
combined_pdp_plot <- wrap_plots(pdp_plots, ncol = length(pdp_plots))  # ncol set to the number of plots for horizontal arrangement

# Save the combined plot as a PDF with appropriate width and height
ggsave("pdp_Biom2_comparison_rf_xgb.pdf", combined_pdp_plot, width = 14, height = 4)

# Inform the user
cat("PDF with horizontal PDP comparison plots created and saved as 'pdp_Biom2_comparison_rf_xgb.pdf'.\n")

#Sub analysis (this is considered later after the analsis of the entire region)
#the data for lakes were not sufficent to produce output for the variable importance
#Wetland analysis
#Select the data
dataset<- dataset[dataset$Water_body ==2,]#Wetland


#..........................end....................................................




