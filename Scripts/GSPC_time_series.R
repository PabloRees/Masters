pacman::p_load(stats, dplyr, urca, tidyverse ,ggplot2, fixest)
pacman::p_load(fixest)

library(stats) 
library(dplyr)
library(urca)
library(tidyverse)
library(ggplot2)

MasterGSPC <- read.csv("/Users/pablo/Desktop/Masters /Github Repository/Masters/Data/GSPC.csv" , header = TRUE)

#Mutate date into date format
GSPC <- MasterGSPC %>% mutate(Date = as.Date(Date, format ="%Y-%m-%d" ))

#create % change variable
GSPC <- GSPC %>% mutate(DailyChangePercent = (Close-Open)*100/Open) %>%
  mutate(DailyChangeLog = log(Close)-log(Open)) %>%
  mutate(DailyChange = (Close-Open)) %>% 
  mutate(NormalizedDailyChange = ((Close-Open) - mean(Close-Open))/sd(Close-Open))

GSPCafter1982 <- GSPC %>% dplyr::filter(Date > as.Date("1982-04-19"))

GSPCafter1982 <- GSPCafter1982 %>% mutate(Weekday = weekdays(as.Date(Date))) %>%
  mutate(Month = months(as.Date(Date))) %>%
  mutate(Year = format(as.Date(Date),"%Y")) %>%
  mutate(Monthday = format(as.Date(Date),"%d"))

DailyChangePercent <- GSPCafter1982 %>% dplyr::select(DailyChangePercent)

Closing <- GSPCafter1982 %>% dplyr::select(Close)

plot_time_series(DailyChangePercent)

DailyChangePercent

change_acf <- stats::acf(
  x = DailyChangePercent,
  plot = T,
  type = "correlation"
)

change_pacf <- stats::acf(
  x = DailyChangePercent,
  plot = T,
  type = "partial") # we want the PACF option

change_pacf

DailyChangeVector <- DailyChangePercent%>% pull(DailyChangePercent)

change_ADF1 <- ur.df(
  y= DailyChangeVector, #vector
  type = "trend",
  lags = 3,
  selectlags = "AIC")

summary(change_ADF1) #yields 3 significant lags and a p-score less than 0,05 which suggests stationarity

summary(DailyChangePercent)

GSPCafter1982 %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = log(Close)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = log(Open)-1), size = 0.1, alpha = 1, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "LogClose and LogOpen-1", y = "LogPrice", x = "Year")

#The graph also suggests stationarity
GSPCafter1982 %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = DailyChangePercent), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  #geom_line(aes(x = Date, y = DailyChange), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 40)) +
  theme(axis.text.y = element_text( size = 40)) +
  # rotate x axis labels
  labs(title = "Daily Percentage Change", y = "%", x = "Year") + 
  theme(plot.title = element_text(size=40))+
  theme(axis.title = element_text(size=40))


#The graph also suggests stationarity
GSPCafter1982 %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = DailyChangeLog), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  #geom_line(aes(x = Date, y = DailyChange), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Log(Close) - Log(Open)", y = "Log Price Change", x = "Year")

GSPCafter1982 %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = NormalizedDailyChange), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  #geom_line(aes(x = Date, y = DailyChange), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Normalized Daily Change", y = "Normalized Daily Change", x = "Year")

#function to shift the data (x) by n spaces
shift <- function(x, n){
  c(x[-(seq(n))], rep(NA, n))
}

DCP_1 <- shift(DailyChangeVector, 1)
DCP_2 <- shift(DailyChangeVector, 2)
DCP_4 <- shift(DailyChangeVector, 4)
DCP_12 <- shift(DailyChangeVector, 12)
DCP_15 <- shift(DailyChangeVector, 15)
DCP_16 <- shift(DailyChangeVector, 16)
DCP_18 <- shift(DailyChangeVector, 18)
DCP_27 <- shift(DailyChangeVector, 27)
DCP_32 <- shift(DailyChangeVector, 32)
DCP_34 <- shift(DailyChangeVector, 34)

stdVol <-  (GSPCafter1982$Volume - mean(GSPCafter1982$Volume))/sd(GSPCafter1982$Volume)
stdVol
stdVol_1 <- shift(stdVol, 1)
stdVol_2 <- shift(stdVol, 2)
stdVol_3 <- shift(stdVol, 3)

DailyChangeAR_df <- GSPCafter1982 %>% dplyr::select(Date, Weekday, Monthday, Month, Year) %>%
  data.frame(DailyChangeVector, stdVol_1, stdVol_2, stdVol_3, DCP_1, DCP_2, DCP_4, DCP_12, DCP_15, DCP_16, DCP_18, DCP_27, DCP_32, DCP_34)

DailyChangeAR_df %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = stdVol), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  #geom_line(aes(x = Date, y = DailyChange), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Standardized Volume", y = "stdVol ($)", x = "Year")

DailyChangeAR_df %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = stdVol, y = DailyChangeVector), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Standardized Volume vs % Daily Change", x = "stdVol ($)", y = "% Daily Change")

library(fixest)
  
#         regression (dependent ~ independent + independent, data = dataframe)
DailyChangeAR_full <- feols(DailyChangeVector ~ DCP_1 + DCP_2 + DCP_4 + DCP_12 + DCP_15 + DCP_16 + DCP_18 + DCP_27 + DCP_32 + DCP_34, data = DailyChangeAR_df) # Adding an explanatory continuous variable

summary(DailyChangeAR_full) 

#         dropping DCP's with * or ** significance - Adj R2 got worse
DailyChangeAR_highSig <- feols(DailyChangeVector ~ DCP_1 + DCP_2 +  DCP_12 + DCP_15 + DCP_16    + DCP_34, data = DailyChangeAR_df) # Adding an explanatory continuous variable

summary(DailyChangeAR_highSig) 

vol_reg <-  feols(DailyChangeVector ~  stdVol_1 ,data=DailyChangeAR_df)
summary(vol_reg)

dayRegression <- feols(DailyChangePercent ~ i(Weekday) + i(Monthday) + i(Month) + i(Year), data = GSPCafter1982)
summary(dayRegression)

DailyChangeAR_df <- DailyChangeAR_df %>% mutate(September = ifelse(Month == 'September',1,0), KK1 = ifelse(Year == 2001,1,0), KK2 = ifelse(Year == 2002,1,0), KK8 = ifelse(Year == 2008,1,0), KK18 = ifelse(Year == 2018,1,0)) 

DailyChangeAllSigModel  <- feols(DailyChangeVector ~ stdVol_1 + DCP_1 + DCP_2 + DCP_4 + DCP_12 + DCP_15 + DCP_16 + DCP_18 + DCP_27 + DCP_32 + DCP_34  + i(Monthday) + i(September) + i(KK1) + i(KK2) +i(KK8) +i(KK18), data = DailyChangeAR_df) # Adding an explanatory continuous variable

summary(DailyChangeAllSigModel)

