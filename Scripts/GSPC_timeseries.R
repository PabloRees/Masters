rm(list=ls()) #Removes all items in Environment!
graphics.off()

pacman::p_load(stats, dplyr, urca, tidyverse ,ggplot2, fixest, FinTS, fGarch, tsm)
pacman::p_load_gh("r-lib/devtools", "KevinKotze/tsm")

library(devtools)
library(stats)
library(dplyr)
library(urca)
library(tidyverse)
library(ggplot2)
library(fixest)
library(FinTS)
library(fGarch)
library(tsm)
library(tseries)

source('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Scripts/std_fin-ts_data_setup.R')
source('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Scripts/GSPC_timeseries_functions.R')

##
## DATA SETUP
GSPC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv" , header = T) #this should load from a sql database rather

GSPC <- stdDataSetup(GSPC,volFilter = T)

GSPC <- blackSwan(GSPC,3)
GSPC <- blackSwan(GSPC,4)
GSPC <- blackSwan(GSPC,5)

start <- c(format(first(GSPC$Date),"%Y"),format(first(GSPC$Date),"%d"))
freq <- abs(round(length(GSPC$Date)/(as.integer(format(last(GSPC$Date),"%Y")) - as.integer(format(first(GSPC$Date),"%Y")))))


logDif <-  ts(GSPC$logDif , start=start , freq = freq)
rm(freq,start)


#######################################
##Auto Correlation and Dickey Fuller Tests

acf <- stats::acf(
  x = logDif,
  plot = T,
  type = "correlation"
)


pacf <- stats::acf(
  x = logDif,
  plot = T,
  type = "partial") # we want the PACF option

library(urca)
ADF1 <- ur.df(
  y= logDif, #vector
  type = "trend",
  lags = 3,
  selectlags = "AIC")

summary(ADF1) #yields 1 significant lag and a p-score less than 0,05 which suggests stationarity

rm(acf, pacf, ADF1)

#####################################
####Volume analysis

volumeAnalysis(GSPC)

####################################
##ARCH  TESTS - Kevin Kotze tutorial

DlogDif_df <- ARCHTests(logDif)[13]

#####################################
###Positive vs negative movements

positive_vec <- vector()
negative_vec <- vector()

DlogDif <- logDif-mean(logDif)


for (i in DlogDif){
  if (i >0){
    positive_vec <- c(positive_vec, i)}

  else{negative_vec <- c(negative_vec, i)}

}

pos_neg_ratio <- mean(negative_vec)/mean(positive_vec)

pos_neg_ratio

pos_neg_transform <- vector()

for (i in DlogDif){
  if (i >0){
    pos_neg_transform <- c(pos_neg_transform, 1)}

  else{pos_neg_transform <- c(pos_neg_transform, pos_neg_ratio)}

}

pos_neg_transform <- shift(pos_neg_transform,1)

rm(negative_vec,positive_vec,i,pos_neg_ratio)
######################################


#######################################
##Export data setup

DlogDif_df <- subset(DlogDif_df, select = c(DlogDif,DlogDif_1,DlogDif_2))
DlogDif_df <- DlogDif_df |> mutate(absDlogDif = abs(DlogDif),absDlogDif_1 = abs(DlogDif_1))

output_df <- data.frame(Date = GSPC$Date, DlogDif_df,logDif = GSPC$logDif,
                        logDif_date_resid = GSPC$logDif_date_resid,
                        logDif_date_resid_1 = GSPC$logDif_date_resid_1,
                        logDif_date_resid_2 = GSPC$logDif_date_resid_2,

                        blackSwan_SD3_1 = shift(GSPC$blackSwan_SD3,1),
                        blackSwan_SD4_1 = shift(GSPC$blackSwan_SD4,1),
                        blackSwan_SD5_1 = shift(GSPC$blackSwan_SD5,1),
                        stdVol_1DateResid = GSPC$stdVol_1DateResid,
                        pos_neg_transform)

rm(GSPC,DlogDif_df,DlogDif,pos_neg_transform)

funreg <- feols(DlogDif~ i(blackSwan_SD5_1) + i(pos_neg_transform) ,data = output_df)
summary(funreg)

write.csv(output_df,"/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/GSPC_features.csv", row.names = FALSE)

sd(output_df$logDif)

rm(list=ls()) #Removes all items in Environment!
graphics.off()

