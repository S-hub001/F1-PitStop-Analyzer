# Load libraries
library(stringr)
library(tidyverse)
library(lubridate)
library(janitor)
library(hms)
library(ggplot2)

# Load data
pitstops <- read_csv("C:/Users/HP/Downloads/f1_pitstops_2018_2024.csv")

# 1. Clean column names
pitstops <- pitstops %>% clean_names()

# 2. Remove duplicates
pitstops <- distinct(pitstops)

# 3. Fix "Final Stint" and convert to numeric safely
pitstops <- pitstops %>%
  mutate(
    pit_time = ifelse(str_to_lower(pit_time) == "final stint", NA, pit_time),
    pit_time = as.numeric(pit_time)
  )

# 4. Convert data types
pitstops <- pitstops %>%
  mutate(
    season = as.integer(season),
    round = as.integer(round),
    date = dmy(date),
    time_of_race = hms::as_hms(time_of_race),
    air_temp_c = as.numeric(air_temp_c),
    track_temp_c = as.numeric(track_temp_c),
    humidity_percent = as.numeric(humidity_percent),
    wind_speed_kmh = as.numeric(wind_speed_kmh),
    avg_pit_stop_time = as.numeric(avg_pit_stop_time),
    stint_length = as.integer(stint_length),
    position_changes = as.numeric(position_changes),
    tire_usage_aggression = as.numeric(tire_usage_aggression),
    driver_aggression_score = as.numeric(driver_aggression_score),
    fast_lap_attempts = as.numeric(fast_lap_attempts),
    lap_time_variation = as.numeric(lap_time_variation)
  )

# 5. Fix encoding issues (Kimi style)
pitstops$driver <- str_replace_all(pitstops$driver, "Ã[^\\s]*", "ä")

# 6. Drop duplicate column if it exists
if ("total_pit_stops_2" %in% names(pitstops)) {
  pitstops <- pitstops %>% select(-total_pit_stops_2)
}

# 7. Drop rows with critical environmental missing values
pitstops <- pitstops %>%
  drop_na(date, time_of_race, air_temp_c, track_temp_c, humidity_percent)

# 8. Impute remaining missing values
pitstops <- pitstops %>%
  mutate(
    pit_time = replace_na(pit_time, 0),
    pit_lap = replace_na(pit_lap, 0),
    avg_pit_stop_time = replace_na(avg_pit_stop_time, 0),
    stint_length = replace_na(stint_length, 0),
    stint = replace_na(stint, 0),
    tire_compound = replace_na(tire_compound, "Unknown"),
    tire_usage_aggression = replace_na(tire_usage_aggression, median(tire_usage_aggression, na.rm = TRUE)),
    fast_lap_attempts = replace_na(fast_lap_attempts, 0),
    driver_aggression_score = replace_na(driver_aggression_score, median(driver_aggression_score, na.rm = TRUE)),
    lap_time_variation = replace_na(lap_time_variation, median(lap_time_variation, na.rm = TRUE))
  )

# 9. Check missing values summary
missing_summary <- colSums(is.na(pitstops))
print(missing_summary)

# check dimensions
dim(pitstops)


# 10. Save clean version
write_csv(pitstops, "cleaned_f1_pitstops.csv")

############ Basic Visualizations ##############

pitstops <- read_csv("C:/Users/HP/Downloads/cleaned_f1_pitstops.csv")  # Assuming you saved it earlier


# 1. Distribution of Pit Stop Time
ggplot(pitstops, aes(x = pit_time)) +
  geom_histogram(fill = "darkred", bins = 30, alpha = 0.8) +
  labs(title = "Distribution of Pit Stop Time", x = "Pit Time (sec)", y = "Frequency")

# 2. Driver Aggression vs Position Change
ggplot(pitstops, aes(x = driver_aggression_score, y = position_changes)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", color = "darkgreen", se = FALSE) +
  labs(title = "Aggression Score vs Position Changes", x = "Aggression Score", y = "Position Change")

# 3. Pit Time vs Final Position
ggplot(pitstops, aes(x = pit_time, y = position)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(title = "Pit Stop Time vs Race Position", x = "Pit Time (sec)", y = "Final Position (Lower is Better)")

# 4. Average Pit Stop Time by Team
pitstops %>%
  group_by(constructor) %>%
  summarise(mean_pit = mean(pit_time, na.rm = TRUE)) %>%
  arrange(mean_pit) %>%
  ggplot(aes(x = reorder(constructor, mean_pit), y = mean_pit)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(title = "Average Pit Stop Time by Constructor", x = "Constructor", y = "Mean Pit Time (sec)")

# 5. Heatmap: Feature Correlation
library(corrplot)

numeric_cols <- pitstops %>%
  select(where(is.numeric)) %>%
  na.omit()  # corrplot doesn’t like NAs

cor_matrix <- cor(numeric_cols)

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.8, number.cex = 0.7, title = "Feature Correlation Heatmap")


# 6. Driver wise aggression
pitstops %>%
  group_by(driver) %>%
  summarise(avg_aggression = mean(driver_aggression_score, na.rm = TRUE)) %>%
  arrange(desc(avg_aggression)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(driver, avg_aggression), y = avg_aggression)) +
  geom_col(fill = "tomato") +
  coord_flip() +
  labs(title = "Top 10 Most Aggressive Drivers", x = "Driver", y = "Avg Aggression Score")


# 7. Fast Lap Attempts vs Final Position
library(ggplot2)
library(hexbin)

ggplot(pitstops, aes(x = fast_lap_attempts, y = position)) +
  stat_binhex(bins = 25, aes(fill = ..count..)) +
  scale_fill_gradient(low = "skyblue", high = "darkblue") +
  labs(
    title = "Heatmap of Fast Lap Attempts vs Final Position",
    x = "Fast Lap Attempts",
    y = "Final Position",
    fill = "Driver Count"
  ) +
  theme_minimal()


colnames(pitstops)
# Load ML libraries
library(caret)
library(e1071)       # Naive Bayes
library(randomForest)
library(class)       # KNN
library(glmnet)      # Regularized Logistic Regression

# Step 1: Create binary target & select relevant features
pitstops <- pitstops %>%
  mutate(top5 = ifelse(position <= 5, 1, 0))

ml_data <- pitstops %>%
  select(top5, driver_aggression_score, pit_time, avg_pit_stop_time,
         tire_usage_aggression, fast_lap_attempts, position_changes) %>%
  drop_na()

# Step 2: Train/Test Split
set.seed(123)
train_index <- createDataPartition(ml_data$top5, p = 0.8, list = FALSE)
train_data <- ml_data[train_index, ]
test_data <- ml_data[-train_index, ]

# ---------- ✅ Step 3: Regularized Logistic Regression with glmnet ----------
# Prepare matrices for glmnet
x <- as.matrix(train_data[, -1])
y <- train_data$top5
x_test <- as.matrix(test_data[, -1])

# Train logistic regression with Ridge penalty (alpha = 0)
log_model_ridge <- glmnet(x, y, family = "binomial", alpha = 0)

# Predict
log_pred <- predict(log_model_ridge, newx = x_test, s = 0.01, type = "response")
log_class <- ifelse(log_pred > 0.5, 1, 0)

# Step 4: Random Forest
rf_model <- randomForest(as.factor(top5) ~ ., data = train_data)
rf_pred <- predict(rf_model, newdata = test_data)

# Step 5: KNN (requires scaled numeric features)
train_scaled <- scale(train_data[,-1])
test_scaled <- scale(test_data[,-1],
                     center = attr(train_scaled, "scaled:center"),
                     scale = attr(train_scaled, "scaled:scale"))
knn_pred <- knn(train = train_scaled, test = test_scaled,
                cl = train_data$top5, k = 5)

# Step 6: Naive Bayes
nb_model <- naiveBayes(as.factor(top5) ~ ., data = train_data)
nb_pred <- predict(nb_model, newdata = test_data)

# ------------------ Evaluation ------------------

conf_matrix <- function(true, pred, model) {
  cat("\nModel:", model, "\n")
  print(confusionMatrix(as.factor(pred), as.factor(true), positive = "1"))
}

# Print all evaluations
conf_matrix(test_data$top5, log_class, "Logistic Regression (Ridge)")
conf_matrix(test_data$top5, rf_pred, "Random Forest")
conf_matrix(test_data$top5, knn_pred, "KNN")
conf_matrix(test_data$top5, nb_pred, "Naive Bayes")

######### F1 Score Bar Chart #########

library(ggplot2)

# F1 scores for each model
f1_scores <- data.frame(
  Model = c("Logistic (Ridge)", "Random Forest", "KNN", "Naive Bayes"),
  F1_Score = c(0.9986, 1.0000, 0.9893, 0.9879)
)

# Bar chart with zoomed-in y-axis for clearer difference
ggplot(f1_scores, aes(x = Model, y = F1_Score, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(F1_Score, 4)), vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c(
    "Logistic (Ridge)" = "#B1380B",
    "Random Forest" = "#731F00",
    "KNN" = "#F0561D",
    "Naive Bayes" = "#F89B78"
  )) +
  labs(
    title = "F1 Score Comparison of ML Models",
    x = "Model",
    y = "F1 Score"
  ) +
  coord_cartesian(ylim = c(0.97, 1.0011)) +  # Zoomed-in y-axis
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1),
    legend.position = "none"
  )


# Final F1 Score Table
f1_scores <- data.frame(
  Model = c("Logistic Regression (Ridge)", "Random Forest", "KNN", "Naive Bayes"),
  F1_Score = c(0.9986, 1.0000, 0.9893, 0.9879),
  Accuracy = c(0.9986, 1.0000, 0.9893, 0.9879)
)

print(f1_scores)


# Confusion matrix
library(caret)
library(ggplot2)
library(gridExtra)

# Helper function to generate confusion matrix plot
plot_cm <- function(pred, truth, model_name, fill_color = "#FFA726") {
  cm <- confusionMatrix(as.factor(pred), as.factor(truth), positive = "1")
  cm_df <- as.data.frame(as.table(cm$table))
  colnames(cm_df) <- c("Prediction", "Reference", "Freq")
  
  p <- ggplot(data = cm_df, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Freq), color = "white") +
    scale_fill_gradient(low = "#FFF3E0", high = fill_color) +
    geom_text(aes(label = Freq), size = 5, fontface = "bold") +
    labs(title = paste("Confusion Matrix -", model_name), x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", size = 14))
  
  return(p)
}

# Create confusion matrix plots for all models
cm_log <- plot_cm(log_class, test_data$top5, "Logistic Regression", "#EF5350")
cm_rf  <- plot_cm(rf_pred, test_data$top5, "Random Forest", "#66BB6A")
cm_knn <- plot_cm(knn_pred, test_data$top5, "KNN", "#42A5F5")
cm_nb  <- plot_cm(nb_pred, test_data$top5, "Naive Bayes", "#AB47BC")

# Arrange all in one window
grid.arrange(cm_log, cm_rf, cm_knn, cm_nb, ncol = 2)






