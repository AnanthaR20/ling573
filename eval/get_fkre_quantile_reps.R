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

write.csv(t2_gold,file = "gold_fkre_quantiles.csv",row.names = F)
write.csv(t2_gen,file = "gen_fkre_quantiles.csv",row.names = F)



# get samples of

a <- read.csv("output_rows_where_gold_tword_above250.csv")
a_quantiles <- a %>% mutate(fkre_quantile.GOLD = ntile(fkre.GOLD,5),fkre_quantile.GEN = ntile(fkre.GEN,5))
# View(a_quantiles)
gold_quants <- c()
gen_quants <- c()
for(i in 1:5){
     df_gold <- a_quantiles %>% filter(fkre_quantile.GOLD == i)
     df_gen <- a_quantiles %>% filter(fkre_quantile.GEN == i)
     gold_quants <- c(gold_quants,sample(df_gold$X,1))
     gen_quants <- c(gen_quants,sample(df_gen$X,1))
}

a_quantiles %>% filter(X %in% gold_quants) %>% 
  write.csv(file = "samples_where_gold_tword_above250.csv")

# a_quantiles %>% filter(X %in% gen_quants) %>% 
#   write.csv(file = "gen_quantile_samples_where_gold_tword_above250.csv")



