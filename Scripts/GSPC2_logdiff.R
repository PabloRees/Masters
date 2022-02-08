rm(list=ls()) #Removes all items in Environment!

pacman::p_load(stats, dplyr, urca, tidyverse ,ggplot2, fixest, FinTS)
library(stats)
library(dplyr)
library(urca)
library(tidyverse)
library(ggplot2)
library(fixest)
library(FinTS)

#shift function
shift <- function(x, n){
  c(x[-(seq(n))], rep(NA, n))
}

MasterGSPC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv" , header = TRUE)

WorkingGSPC <- MasterGSPC %>% mutate(logClose = log(Close), Date = as.Date(Date, format ="%Y-%m-%d" )) %>%
  filter(Date>as.Date('1949-12-30'))

stdVol = (WorkingGSPC$Volume - mean(WorkingGSPC$Volume))/sd(WorkingGSPC$Volume)

diffLogClose <- diff(WorkingGSPC$logClose)

WorkingGSPC = WorkingGSPC[-1,]

stdVol = data.frame(stdVol)

WorkingGSPC <- WorkingGSPC%>% data.frame(diffLogClose,stdVol[-1,]) %>%
  filter()

  
acf <- stats::acf(
  x = diffLogClose,
  plot = T,
  type = "correlation"
)


pacf <- stats::acf(
  x = diffLogClose,
  plot = T,
  type = "partial") # we want the PACF option


ADF1 <- ur.df(
  y= diffLogClose, #vector
  type = "trend",
  lags = 3,
  selectlags = "AIC")

summary(ADF1) #yields 1 significant lags and a p-score less than 0,05 which suggests stationarity

CTS <- ts(WorkingGSPC$diffLogClose)
hist(CTS, main="", breaks=50, freq=FALSE, col="darkgreen")

dpcARCH_test <- FinTS::ArchTest(CTS, lags = 1, demean = TRUE)
dpcARCH_test

CGARCH_test <- garch(CTS)
summary(CGARCH_test)

CARCH_test2 <- garch(CTS,c(0,1))
summary(CARCH_test2)

hhat <- ts(2*CARCH_test2$fitted.values[-1,1]^2)
plot.ts(hhat)


