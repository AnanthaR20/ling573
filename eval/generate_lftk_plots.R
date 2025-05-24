# File for generating model analysis plots for ling 573 project
# File Arguments:
MODEL_NAME <- "PegasusBillSum_with_se3_t5_toy" # The name of the Model whose output is being evaluated
MODEL_OUTPUT_PATH <- "../output/deliverable_3/pegasusbillsum_clean_se3_t5_simple_toy.csv" # place to look for model output csv
GOLD_PATH <- "gold_lftk.csv" # place to look for gold data csv
ANALYSIS_PATH <- "deliverable_3/" # place to write the plots and tests to. Must end with a "/"
# ----------------------------------------------------------------------- #
# Do not change these variables. These represent suffixes we expect to read in from csv headers
GOLD_SUFFIX <- ".GOLD"
GEN_SUFFIX <- ".GEN"
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ------------------------- (0) Setup Variables ------------------------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
library(tidyr)
library(ggplot2)
library(stringr)
library(dplyr)

# LFTK readability and other metrics of gold and generated summaries on test partition
# These two CSVs must have to same columns
lftk <- list()
lftk[["gold"]] <- read.csv(GOLD_PATH)
lftk[['gen']] <- read.csv(MODEL_OUTPUT_PATH)

# Remove the redundant '.GOLD' lftk columns if included in file headers at MODEL_OUTPUT_PATH then
# set column names properly for processing
lftk$gen <- lftk$gen %>% select(-ends_with(GOLD_SUFFIX))
colnames(lftk$gold) <- str_replace(colnames(lftk$gold),GOLD_SUFFIX,"")
colnames(lftk$gen) <- str_replace(colnames(lftk$gen),GEN_SUFFIX,"")

# Refers to LFTK family of metrics. See README
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

# ------------------------------------------------------------ #
# -------------------- (1) Generate plots -------------------- #
# ------------------------------------------------------------ #

for (feature in colnames(lftk$gen)) {
  for(fam in names(family)){
    if(!(feature %in% family[[fam]])){
      next
    }
    plt <- 
    ggplot() + 
      geom_histogram(aes(x=lftk$gold[[feature]],fill='Gold'),bins=60) +
      geom_histogram(aes(x=lftk$gen[[feature]], fill='Generated'), alpha=0.55,bins=60) +
      # scale_x_continuous(breaks = seq(0,30,5)) +
      labs(x = 'Value', y = "Count", title= str_c(feature," for ",MODEL_NAME," and Gold Summaries")) +
      scale_color_manual(name='Legend',
                         breaks=c("Gold","Generated"),
                         values = c("Gold"="gold",'Generated'="black"))
    
    # Save plots in right place
    path_to_save <- str_c(ANALYSIS_PATH,"lftk_plots/",fam,"/")
    ggsave(str_c(path_to_save,feature,"_distribution_gen_and_gold_summaries.png"),plt,create.dir = T)
    
  } # ----- End of Attributes loop
} # ----- End of lftk features loop


# -------------------------------------------------------------------------- #
# ---------------- (2) Run t.tests for various LFTK metrics ---------------- #
# -------------------------------------------------------------------------- #

# Prep directories for populating analysis
feature_to_batch = "t_char"
path_to_write = str_c(ANALYSIS_PATH,"lftk_tests/t_tests/grouped_by=",feature_to_batch,"/")

unlink(str_c(ANALYSIS_PATH,"lftk_tests/t_tests"),recursive = T)
dir.create(str_c(ANALYSIS_PATH,"lftk_tests/"),showWarnings = F)
dir.create(str_c(ANALYSIS_PATH,"lftk_tests/t_tests"),showWarnings = F)
dir.create(path_to_write,showWarnings = F)


# Generate quantile column for this feature
num_quantiles = 5
gold_quantiles <- lftk$gold %>% mutate(quantile = ntile(t_char,num_quantiles))
gen_quantiles <- lftk$gen %>% mutate(quantile = ntile(t_char,num_quantiles))

# Get readformula t tests
for(feature in family$readformula){
  # Do t tests for each quantile
  for(q in 1:num_quantiles){
    gold_subset <- gold_quantiles %>% filter(quantile == q)
    gen_subset <- gen_quantiles %>% filter(quantile == q)
    
    result <- capture.output(t.test(gen_subset[[feature]],gold_subset[[feature]]))
    
    # write to file
    write(
      c(str_c("--- LFTK feature = ",feature,": ", MODEL_NAME," vs Gold Summaries - Quantile ", q, " of ", num_quantiles, " - Data batched by character count"), result),
      file = str_c(path_to_write,feature,"_by_groups.txt"),
      append = T
    )
    
  }
}

# Get wordsent t tests
for(feature in family$wordsent){
  
  # result <- capture.output(t.test(gen_subset[[feature]],gold_subset[[feature]]))
  result <- capture.output(t.test(lftk$gen[[feature]],lftk$gold[[feature]]))
  
  # write to file
  write(
    c(str_c("--- LFTK feature = ",feature,": ",MODEL_NAME," vs Gold Summaries - full data"), result),
    file = str_c(ANALYSIS_PATH,"lftk_tests/t_tests/wordsent_gen_vs_gold.txt"),
    append = T
  )
}

# Get full data readformula t tests
for(feature in family$readformula){
  
  # result <- capture.output(t.test(gen_subset[[feature]],gold_subset[[feature]]))
  result <- capture.output(t.test(lftk$gen[[feature]],lftk$gold[[feature]]))
  
  # write to file
  write(
    c(str_c("--- LFTK feature = ",feature,": ",MODEL_NAME," vs Gold Summaries - full data"), result),
    file = str_c(ANALYSIS_PATH,"lftk_tests/t_tests/readformula_gen_vs_gold.txt"),
    append = T
  )
}


# ----------------------------------------------------------------------- #
# ---------- (3) Generate Samples for Qualitative Analysis of fkre ------ #
# ----------------------------------------------------------------------- #

# Add quantile information to dataframes for filtering
set.seed(20)
num_quantiles = 5
gold_quantiles <- lftk$gold %>% mutate(quantile = ntile(fkre,num_quantiles))
gen_quantiles <- lftk$gen %>% mutate(quantile = ntile(fkre,num_quantiles))

# Make a list of fkre quantiles for gold summaries fkre and model output fkre
smps_gold <- c()
smps_gen <- c()
for(i in 1:num_quantiles){
  df_gold <- gold_quantiles %>% filter(quantile == i)
  df_gen <- gen_quantiles %>% filter(quantile == i)
  smps_gold <- c(smps_gold,sample(df_gold$X,1))
  smps_gen <- c(smps_gen,sample(df_gen$X,1))
}

t_gold <- gold_quantiles %>% filter(X %in% smps_gold)
t2_gold <- t_gold %>% left_join(gen_quantiles,by="X",suffix = c(".GOLD",".GEN"))

t_gen <- gen_quantiles %>% filter(X %in% smps_gen)
t2_gen <- t_gen %>% left_join(gold_quantiles,by="X",suffix = c(".GEN",".GOLD"))


# Prep directories for writing analysis to files
unlink(str_c(ANALYSIS_PATH,"lftk_qa"),recursive = T)
dir.create(str_c(ANALYSIS_PATH,"lftk_qa"),showWarnings = F)

write.csv(t2_gold,file = str_c(ANALYSIS_PATH,"lftk_qa/fkre_quantiles_gold.csv"),row.names = F)
write.csv(t2_gen,file = str_c(ANALYSIS_PATH,"lftk_qa/fkre_quantiles_",MODEL_NAME,".csv"),row.names = F)






# Old plots I may want to use later

# Generate bar charts for readability metrics
# j <- lftk$gold %>% left_join(lftk$gen,by="X",suffix=c(".GOLD",".GEN"))
# bar <- j %>% pivot_longer(cols = colnames(j)[str_detect(colnames(j),"[(GOLD)(GEN)]")])
# bar <- bar %>% mutate(metric = str_extract(name,"([^.]*)\\.",group=1),
#                       summary_type = str_extract(name,"\\.(.*)",group=1))
# 
# bar %>% filter(metric %in% family$readformula) %>%
#   ggplot(aes(x = summary_type,y=value)) +
#   geom_boxplot() +
#   facet_wrap(~metric) +
#   coord_cartesian(ylim = c(-50, 100)) +
#   labs(x = "Summary Type",
#        y = "Value",
#        title = "Generated vs Gold Summaries")



