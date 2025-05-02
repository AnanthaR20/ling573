# Anantha Rao 
# File for generating analysis plots for ling 573 project
# This file expects to be in the same directory as "gold_lftk.csv" and "gen_lftk.csv"
# Both of these files must have the same columns. 
library(tidyr)
library(ggplot2)
library(stringr)
library(dplyr)

# LFTK readability and other metrics of gold and generated summaries on test partition
# gold_lftk.csv and gen_lftk.csv must have to same columns
lftk <- list()
lftk[["gold"]] <- read.csv("gold_lftk.csv")
lftk[['gen']] <- read.csv("gen_lftk.csv")

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

# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #
# ---------- Everything Above is Setup. Below Generates Plots ----------- #
# ----------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #

# (1) Generate plots
for (feature in colnames(lftk$gold)) {
  for(fam in names(family)){
    if(!(feature %in% family[[fam]])){
      next
    }
    plt <- 
    ggplot() + 
      geom_histogram(aes(x=lftk$gold[[feature]],fill='Gold'),bins=60) +
      geom_histogram(aes(x=lftk$gen[[feature]], fill='Generated'), alpha=0.55,bins=60) +
      # scale_x_continuous(breaks = seq(0,30,5)) +
      labs(x = 'Value', y = "Count", title= str_c(feature," for Generated and Gold Summaries")) +
      scale_color_manual(name='Legend',
                         breaks=c("Gold","Generated"),
                         values = c("Gold"="gold",'Generated'="black"))
    
    # Save plots in right place
    path_to_save <- str_c("lftk_plots/",fam,"/")
    ggsave(str_c(path_to_save,feature,"_distribution_gen_and_gold_summaries.png"),plt,create.dir = T)
    
  } # ----- End of Attributes loop
} # ----- End of Generate Plots


# (2) Run t.tests and save results
# Generate quantile column for this feature
num_quantiles = 5
gold_quantiles <- lftk$gold %>% mutate(quantile = ntile(t_char,num_quantiles))
gen_quantiles <- lftk$gen %>% mutate(quantile = ntile(t_char,num_quantiles))
feature_to_batch = "t_char"

# Prep directories
unlink("lftk_tests/",recursive = T)
path_to_write = str_c("lftk_tests/grouped_by=",feature_to_batch,"/")
dir.create("lftk_tests/",showWarnings = F)
dir.create(path_to_write,showWarnings = F)

# Generate readformula t tests
for(feature in family$readformula){
  # Do t tests for each quantile
  for(q in 1:num_quantiles){
    gold_subset <- gold_quantiles %>% filter(quantile == q)
    gen_subset <- gen_quantiles %>% filter(quantile == q)
    
    result <- capture.output(t.test(gen_subset[[feature]],gold_subset[[feature]]))
    
    # write to file
    write(
      c(str_c("--- LFTK feature=",feature,": Generated vs Gold Summaries - Quantile ", q, " of ", num_quantiles, " - Data batched by character count"), result),
      file = str_c(path_to_write,feature,"_by_groups.txt"),
      append = T
    )
    
  }
}

# (3) Get basic t tests for wordsent metrics
# Generate t tests
for(feature in family$wordsent){
  
  result <- capture.output(t.test(gen_subset[[feature]],gold_subset[[feature]]))
  
  # write to file
  write(
    c(str_c("--- LFTK feature=",feature,": Generated vs Gold Summaries - full data"), result),
    file = str_c("lftk_tests/wordsent_gen_vs_gold.txt"),
    append = T
  )
  
}


# a <- capture.output(t.test(lftk$gold$fkre,lftk$gen$fkre))
# write(c("fkre-gold-gen7",a),file = "t.test.yoyoyo",append = T)




