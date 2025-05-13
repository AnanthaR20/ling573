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
set.seed(20)
num_quantiles = 5
gold_quantiles <- lftk$gold %>% mutate(quantile = ntile(fkre,num_quantiles))
gen_quantiles <- lftk$gen %>% mutate(quantile = ntile(fkre,num_quantiles))

smps <- c()
for(i in 1:num_quantiles){
  df_gold <- gold_quantiles %>% filter(quantile == i)
  smps <- c(smps,sample(df_gold$X,1))
}

t <- gold_quantiles %>% filter(X %in% smps)
t2 <- t %>% left_join(gen_quantiles,by="X",suffix = c(".GOLD",".GEN"))

write.csv(t2,file = "output_rows_for_5_fkre_quantiles.csv",row.names = F)



