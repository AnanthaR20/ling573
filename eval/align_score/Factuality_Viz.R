library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(pastecs)

# Get names of the files with desired factuality data
data_file_names <- dir()[c(
  grep("AlignScore",dir())
)]


d <- data.frame()
add_on <- data.frame()
for(file_path in data_file_names){
  t <- read_file(file_path)
  # print(file_path)
  add_on <- data.frame(
      model=str_extract(file_path,"=(.+)\\.txt",group = 1),
      row = as.numeric(str_extract(str_extract_all(t,"<ROW>.+</ROW>")[[1]],"<ROW>(.+)</ROW>",group=1)),
      summary = str_extract(str_extract_all(t,"<SUMMARY>(.|\n)*?</SUMMARY>")[[1]],"<SUMMARY>((.|\n)*?)</SUMMARY>",group=1),
      bill = str_extract(str_extract_all(t,"<BILL>(.|\n)*?</BILL>")[[1]],"<BILL>((.|\n)*?)</BILL>",group=1),
      summary_from = str_extract(str_extract_all(t,"<SUMMARY_FROM>.+</SUMMARY_FROM>")[[1]],"<SUMMARY_FROM>(.+)</SUMMARY_FROM>",group=1),
      bill_from = str_extract(str_extract_all(t,"<BILL_FROM>.+</BILL_FROM>")[[1]],"<BILL_FROM>(.+)</BILL_FROM>",group=1),
      alignscore = as.numeric(str_extract(str_extract_all(t,"<ALIGNSCORE>.+</ALIGNSCORE>")[[1]],"<ALIGNSCORE>(.+)</ALIGNSCORE>",group=1))
  )
  
  if(nrow(d) == 0){
    d <- add_on
  } else {
    d <- d %>% rbind(add_on)
  }
}

# Create graphs comparing values to the gold summary values
for (m in unique(d$model)) {
    gold_mean <- (d %>% filter(model == 'gold'))[['alignscore']] %>% mean(na.rm = T)
    m_mean <- (d %>% filter(model == m))[['alignscore']] %>% mean(na.rm = T)
    
    plt <- 
      d %>%
        filter(model == "gold" | model == m) %>% 
          ggplot() + 
            geom_histogram(aes(x=alignscore,fill=model),bins=15) +
            # scale_x_continuous(breaks = seq(0,30,5)) +
            labs(
              x = 'AlignScore', 
              y = "Count", 
              title= str_c("AlignScore Histogram of Summaries")
            ) +
            geom_vline(xintercept = m_mean) +
            geom_vline(xintercept =  gold_mean,color = 'yellow')
    
    # Save plots in right place
    path_to_save <- str_c("plots/")
    ggsave(str_c(path_to_save,"plot_for_",m,".png"),plt,create.dir = T)
    
    # Get gold/alternate model summary
    result <- capture.output(d %>% filter(model == m) %>% stat.desc())
    
    write(
      c(str_c("--- Model: ",m," ---"), result),
      file = str_c("stats/",m,"_descriptive_stats.txt"),
      append = T
    )
    
    
    
} # ----- End of lftk features loop






