# File for generating tristograms for ling 573 project
# File Arguments:
MODEL_NAME <- "wugwATSS-led(on-unsimp)" # The name of the Model whose output is being evaluated
MODEL_OUTPUT_PATH <- "../output/deliverable_4/wugwATSS-led/eval_on_unsimp.csv" # place to look for model output csv
BASELINE_OUTPUT_PATH <- "../output/deliverable_2/pegasusbillsum_baseline_ALL_metrics.csv"
ALT_NAME <- "Gold" # The name of the Alternate Model to evaluate against (or just "Gold" if comparing to gold data)
ALT_PATH <- "gold_lftk.csv" # place to look for gold data/alternate model data csv
ANALYSIS_PATH <- "deliverable_4/tristograms_temp/" # place to write the plots and stats to. Must end with a "/"
HISTOGRAM_BINS <- 60
# Note about these arguments - "ALT_NAME" and "ALT_PATH" could be the output of a different
# model and not necessarily computed from reference data. It's a bad naming convention, but
# when you want to compare two models, put the second model output in the "ALT_PATH" var and
# name it with "ALT_NAME" accordingly.
# ----------------------------------------------------------------------- #
# Do not change these variables. These represent suffixes we expect to read in from csv headers
library(tidyr)
library(ggplot2)
library(stringr)
library(dplyr)
library(pastecs)
library(effsize)
ALT_SUFFIX <- ".GOLD"
GEN_SUFFIX <- ".GEN"


long <- "(.+)\\(.+\\)"
SHORT_MODEL_NAME <- ifelse(str_detect(MODEL_NAME,long), str_extract(MODEL_NAME,long,group=1),MODEL_NAME)
SHORT_ALT_NAME <- ifelse(str_detect(ALT_NAME,long), str_extract(ALT_NAME,long,group=1),ALT_NAME)
BASE_NAME <- "Baseline"

# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ------------------------- (0) Setup Variables ------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #


# LFTK readability and other metrics of gold and generated summaries on test partition
# These two CSVs must have to same columns
metrics <- list() #store metrics for generated summaries and alternate summaries
metrics[["alt"]] <- read.csv(ALT_PATH)
metrics[['gen']] <- read.csv(MODEL_OUTPUT_PATH)
metrics[['base']] <- read.csv(BASELINE_OUTPUT_PATH)

# Remove the redundant '.GOLD' lftk columns if included in file headers at MODEL_OUTPUT_PATH then
# set column names properly for processing
# metrics$alt <- metrics$alt %>% slice(1:16)
metrics$gen <- metrics$gen %>% select(-ends_with(ALT_SUFFIX))
colnames(metrics$alt) <- str_replace(colnames(metrics$alt),ALT_SUFFIX,"")
colnames(metrics$alt) <- str_replace(colnames(metrics$alt),GEN_SUFFIX,"")
colnames(metrics$gen) <- str_replace(colnames(metrics$gen),GEN_SUFFIX,"")

# Refers to family of metrics. Includes LFTK and ROUGE. See README
family <- list()
family$wordsent <- c(
  "t_word",           
  "t_stopword",
  "t_punct",
  "t_syll",
  "t_syll2",
  "t_syll3",
  "t_uword",          
  "t_sent",
  "t_char"
)
family$readformula <- c(
  "fkre",
  "fkgl",
  "fogi",
  "smog",
  "cole",
  "auto"
)
family$worddiff <- c(
  "t_kup",
  "t_bry",
  "t_subtlex_us_zipf"
)
family$entity <- c(
  "t_n_ent_law"
)
family$rouge <- c(
  "rouge1_precision",
  "rouge2_precision",
  "rougeL_precision",
  "rouge1_recall",
  "rouge2_recall",
  "rougeL_recall",
  "rouge1_fmeasure",
  "rouge2_fmeasure",  
  "rougeL_fmeasure"
)

# ------------------------------------------------------------ #
# -------------------- (1) Generate plots -------------------- #
# ------------------------------------------------------------ #

for (feature in colnames(metrics$gen)) {
  for(fam in names(family)){
    if(!(feature %in% family[[fam]]) || !(feature %in% colnames(metrics$alt))){
      next
    }
    plt <- 
      ggplot() + 
      geom_histogram(aes(x=metrics$gen[[feature]], fill=SHORT_MODEL_NAME), alpha=0.95,bins=HISTOGRAM_BINS) +
      geom_histogram(aes(x=metrics$alt[[feature]], fill=SHORT_ALT_NAME),alpha=0.55,bins=HISTOGRAM_BINS) +
      geom_histogram(aes(x=metrics$alt[[base]], fill=SHORT_ALT_NAME),alpha=0.35,bins=HISTOGRAM_BINS) +
      # scale_x_continuous(breaks = seq(0,30,5)) +
      labs(
        x = 'Value', 
        y = "Count", 
        title= str_c(feature," for ",SHORT_MODEL_NAME," and ",SHORT_ALT_NAME," Summaries")
      ) #+
    plt
    # Save plots in right place
    path_to_save <- str_c(ANALYSIS_PATH,"plots/",fam,"/")
    ggsave(str_c(path_to_save,feature,"_distribution_",MODEL_NAME,"_and_",ALT_NAME,"_summaries.png"), plt, create.dir = TRUE)
    
  } # ----- End of Attributes loop
} # ----- End of lftk features loop