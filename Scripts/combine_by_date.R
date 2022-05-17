rm(list=ls()) #Removes all items in Environment!
graphics.off()

pacman::p_load(tidyverse,lubridate)
library(tidyverse)
library(lubridate)

GSPC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/GSPC_features.csv" , header = T) #this should load from a sql database rather
speeches <- read.csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_lite/Tagged_Raw_Speeches_lite_(84155, 12).tsv',sep = '\t')
metadata <- read.csv('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/metadata.csv',header = T)


metadata <- metadata %>% mutate(Date = as.Date(Date, format ="%Y-%m-%d" ))
GSPC <- GSPC %>% mutate(Date = as.Date(Date, format ="%Y-%m-%d" )) %>% arrange(Date)
speeches <- speeches %>% filter(as.Date(Date)>first(GSPC$Date)) 

date_mat<-strsplit(speeches$Date,split = " ")
mat <- matrix(unlist(date_mat),ncol=2,byrow=T)
times_df <- as.data.frame(mat[,2])
date_mat <- strsplit(as.character(times_df$`mat[, 2]`),split=":")
mat <- matrix(unlist(date_mat),ncol=4,byrow=T)
times_df <- as.vector(mat[,1])

speeches <- speeches %>% mutate(Date = as.Date(Date, format = "%Y-%m-%d")) %>%
  mutate(Time = times_df) %>%  
  mutate(Date = ifelse(weekdays(Date) == "Saturday", Date +2, ifelse(weekdays(Date) == "Sunday", Date +1,ifelse(as.numeric(Time) > 16,Date+1,Date)))) %>% 
  mutate(Date = as.Date(Date, format = "%Y-%m-%d", origin)) 

full_df_1 <- left_join(speeches , GSPC)
full_df <- left_join(full_df_1,metadata)

as.character(nrow(full_df))
row <- as.character(nrow(full_df))

shape <- paste('(',row,',',as.character(ncol(full_df)),')')


filePath <- paste("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/final_dataset",shape,'.csv')

write.csv(full_df,filePath, row.names = FALSE)


