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

#shift function
shift <- function(x, n){
  c(x[-(seq(n))], rep(NA, n))
}

#return function
ret <- function(x) {
  pc <- 100*(x/shift(x,1) - 1)
}

#standardize function
std <- function(x) {
  stan <- (x - mean(x, na.rm=T))/sd(x, na.rm=T)
}

##
## DATA SETUP
GSPC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC_1.csv" , header = T) #this should load from a sql database rather

GSPC <- GSPC %>% dplyr::mutate(Date = as.Date(Date, format ="%Y-%m-%d" )) %>%
  dplyr::filter(Volume > 0) %>% arrange(desc(Date)) %>% 
  mutate(Close_1 = shift(Close,1)) %>% 
  mutate(DPC = ret(Close)) %>% 
  mutate(stdVol = std(Volume)) %>% mutate(stdVol_1 = shift(stdVol,1)) %>% 
  arrange(Date)

GSPC <- GSPC[-1,]

start <- c(format(first(GSPC$Date),"%Y"),format(first(GSPC$Date),"%d"))
freq <- abs(round(length(GSPC$Date)/(as.integer(format(last(GSPC$Date),"%Y")) - as.integer(format(first(GSPC$Date),"%Y")))))
DPC <-  ts(GSPC$DPC , start=start , freq = freq)


#######################################
##Auto Correlation and Dickey Fuller Tests

acf <- stats::acf(
  x = DPC,
  plot = T,
  type = "correlation"
)


pacf <- stats::acf(
  x = DPC,
  plot = T,
  type = "partial") # we want the PACF option

library(urca)
ADF1 <- ur.df(
  y= DPC, #vector
  type = "trend",
  lags = 3,
  selectlags = "AIC")

summary(ADF1) #yields 1 significant lag and a p-score less than 0,05 which suggests stationarity


#####################################
####Volume analysis

#stdVol and Daily%Change plot
GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = stdVol_1), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = abs(DPC)), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Lagged Standardized Volume and DPC", y = "stdVol_1 and DPC", x = "Year")

#Daily%change plot
GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = DPC), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "DPC over time", y = "DPC", x = "Year")

GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = abs(DPC)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Absolute DPC over time", y = "Absolute DPC", x = "Year")

Reg1_stdVol_1_DPC <- feols(DPC ~ stdVol_1, data = GSPC) #shows insignificance of volume as a predictor of DCP 
summary(Reg1_stdVol_1_DPC) #regression 1

Reg2_stdVol_1_absDPC <- feols(abs(DPC) ~ stdVol_1, data = GSPC) #shows high significance of volume as a predictor of absolute DCP
summary(Reg2_stdVol_1_absDPC) #Regression 2

Reg3_stdVol_1_Date <- feols(stdVol_1~Date, data=GSPC) #shows that stdVol_1 is not stationary over time
summary(Reg3_stdVol_1_Date) #Regression 3

GSPC <- GSPC %>% mutate(stdVol_1DateResid = resid(Reg3_stdVol_1_Date)) #includes residuals from reg 3

Reg4_residStdVol_1 <- feols(abs(DPC) ~ stdVol_1DateResid, data=GSPC)
summary(Reg4_residStdVol_1) #Regression 4 shows that the residuals are also significant predictors of absolute DPC

Reg5_date_residStdVol_1 <- feols(stdVol_1DateResid ~ Date, data=GSPC)
summary(Reg5_date_residStdVol_1) #Regression 5 shows that there is no relationship between the residuals and date

GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = stdVol), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "stdVol_DateResid ", y = "DateResid of StdVol and stdVol", x = "Year")

GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = abs(DPC)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "stdVol_DateResid and DPC", y = "stdVolControlled and DPC", x = "Year")

cov(abs(GSPC$DPC),GSPC$stdVol_1DateResid)

####################################
##ARCH  TESTS - Kevin Kotze tutorial

#res1 - looking for serial dependence i.e. significant lags
res1 <- ac(DPC) #indicates that the model for the mean equation may have no persistence or may be an AR(2/3) or MA(2/3)
summary(res1)

Box.test(DPC, lag=10, type = "Ljung") #checking if serial correlation exists - significant so serial correlation exists
#number of entries allows for up to 18 lags according to the strictest metric

#res2 - indicates degree of persistence of volatility 
res2 <- ac(abs(DPC))

#res3 - again indicates persistence but not as good a measure as absolute because a lot of the entries are lower than 1 and are therefore made negligible by the ^2
res3 <- ac(DPC^2)

#t-test indicates that the mean is not = 0
t.test(DPC)

#demeaning the series - i.e. removing the mean
DDPC <- DPC-mean(DPC)

#t-test indicates that the DDPC mean is definitely = 0
t.test(DDPC)

Box.test(abs(DDPC), lag = 10, type = 'Ljung') #- There is still serial correlation

#results indicate that ARCH effects are present
archTest(DDPC,10) # - model the error lags in a normal ts regression (1,2 then 1,2,5,6 then 1,2,5,6,8,9)

DDPC_1 <- shift(DDPC,1) 
DDPC_2 <- shift(DDPC,2)
DDPC_5 <- shift(DDPC,5)
DDPC_6 <- shift(DDPC,6)
DDPC_8 <- shift(DDPC,8)
DDPC_9 <- shift(DDPC,9)

DDPC_df <- data.frame(DDPC,DDPC_1,DDPC_2,DDPC_5,DDPC_6,DDPC_8,DDPC_9) #create a dataframe with all the DDPC lags

lagCheck0 <- feols(DDPC ~ DDPC_1 ,data = DDPC_df)
lagCheck1 <- feols(DDPC ~ DDPC_1 + DDPC_2,data = DDPC_df)
lagCheck2 <- feols(DDPC ~ DDPC_1 + DDPC_2 + DDPC_5 + DDPC_6,data = DDPC_df)
lagCheck3 <- feols(DDPC ~ DDPC_1 + DDPC_2 + DDPC_5 + DDPC_6 + DDPC_8 + DDPC_9,data = DDPC_df)

summary(lagCheck0)
summary(lagCheck1)
summary(lagCheck2)
summary(lagCheck3)
#these regressions indicate significance on the 2nd, 6th, 8th and 9th lags

ArchTest(DDPC, lags = 10, demean = TRUE) #also indicates that ARCH effects are present

model1 <- arima(DDPC, order = c(0,0,1)) #AIC = 50000+ - shouldn't just rely on AIC - use other tests - BIC etc.
model1

model2 <- arima(DDPC, order = c(1, 0, 0)) #AIC = 50000+
model2

check1 <- ac(residuals(model1, standardize = T)) #delayed residuals still exist for model 1
check2 <- ac(residuals(model2, standardize = T)) #delayed residuals still exist for model 2

#Model 1 and 2 are essentially giving the same result
model3 <- garchFit(DDPC ~ arma(1,0) + garch(1,1), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model3
summary(model3) #this is the correct model to use because essentially removes any autocorrelation between its residuals indicating that it has captured all of the relevent information for autocorrelation
check3 <- ac(residuals(model3, standardize = T))
#need to include ar1, and variance1 of DDPC in final data

model3.1 <- garchFit(DDPC ~ arma(0,6) + garch(1,1), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model3.1
summary(model3.1)
check3.1 <- ac(residuals(model3.1, standardize = T))
#should maybe include 6 lags or just AR6 because the 6th lag is significant

model3.2 <- garchFit(DDPC ~ arma(2,0) + garch(1,1), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model3.2
summary(model3.2)
check3.2 <- ac(residuals(model3.2, standardize = T))
#need to include ar1, ar2 and variance1 of DDPC in final data - this performs slightly better than model3

model3.3 <- garchFit(DDPC ~ arma(6,0) + garch(1,1), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model3.3
summary(model3.3)
check3.3 <- ac(residuals(model3.2, standardize = T))

model3.4 <- arima(DDPC, order = c(0,0,6))
check3.4 <- ac(residuals(model3.4, standardize = T))

help(garchFit)

model4 <- garchFit(DDPC ~ garch(1, 1), data = DDPC, cond.dist = "std",  #AIC is 2.35 and suggests a standard AR(1) 
                   trace = F) # the high significance of lag 1 indicates that the AR(1) was better
summary(model4)
check4 <- ac(residuals(model4, standardize = T))

model6 <- garchFit(DDPC ~ arma(1,0) + garch(1,0), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model6

model7 <- garchFit(DDPC ~ garch(1,0), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model7

check5 <- ac(abs(residuals(model3, standardize = T)))

check7 <- ac(residuals(model6, standardize = T))

check8 <- ac(residuals(model7, standardize = T))

par(mfrow = c(1, 1))
plot.ts(residuals(model4, standardize = T))

plot.ts(residuals(model3, standardize =T))

prediction <- predict(model3.2, newdata = wine_test)

ARMA06_resid_1 <- shift(residuals(model3.4, standardize = T),1)

#####################################
###Positive vs negative movements

positive_vec <- vector()
negative_vec <- vector()
  
for (i in DDPC)
{
  if (i >0){
    positive_vec <- c(positive_vec, i)}
  
  else{negative_vec <- c(negative_vec, i)}
  
}

pos_neg_ratio <- mean(negative_vec)/mean(positive_vec)

pos_neg_ratio

pos_neg_transform <- vector()

for (i in DDPC)
{
  if (i >0){
    pos_neg_transform <- c(pos_neg_transform, 1)}
  
  else{pos_neg_transform <- c(pos_neg_transform, pos_neg_ratio)}
  
}

pos_neg_transform <- shift(pos_neg_transform,1)

######################################
##EGARCH

pacman::p_load(rugarch, FinTS, zoo, e1071)
library(rugarch)
library(FinTS)
library(zoo)
library(e1071)

x <-  ugarchspec(variance.model = 
                   list(model="eGARCH", garchOrder=c(1,1)),mean.model = 
                   list(armaOrder=c(0,0)))

model5 <- ugarchfit(spec = x, data = DDPC) 

model5

check5 <- ac(residuals(model5, standardize = T))

#######################################
##Export data setup

output_df <- data.frame(DDPC,Date = GSPC$Date, DPC = GSPC$DPC, stdVol_1DateResid = GSPC$stdVol_1DateResid, DDPC_1, DDPC_2, DDPC_6, ARMA06_resid_1,pos_neg_transform, absDDPC_1 = abs(DDPC_1))
  
funreg <- feols(DDPC~absDDPC_1*pos_neg_transform ,data = output_df)
summary(funreg)


write.csv(output_df,"/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC_features.csv", row.names = FALSE)

 