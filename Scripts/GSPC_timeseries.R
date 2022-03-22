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

#stdVol and Daily%Change plot
GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = stdVol_1), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Lagged Standardized Volume and logDif", y = "stdVol_1 and logDif", x = "Year")

#Daily%change plot
GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = logDif), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "logDif over time", y = "logDif", x = "Year")

GSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Absolute logDif over time", y = "Absolute logDif", x = "Year")

Reg1_stdVol_1_logDif <- feols(logDif ~ stdVol_1, data = GSPC) #shows insignificance of volume as a predictor of logDif 
summary(Reg1_stdVol_1_logDif) #regression 1

Reg2_stdVol_1_abslogDif <- feols(abs(logDif) ~ stdVol_1, data = GSPC) #shows high significance of volume as a predictor of absolute logDif
summary(Reg2_stdVol_1_abslogDif) #Regression 2

Reg3_stdVol_1_Date <- feols(stdVol_1~Date, data=GSPC) #shows that stdVol_1 is not stationary over time
summary(Reg3_stdVol_1_Date) #Regression 3

GSPC <- GSPC %>% mutate(stdVol_1DateResid = resid(Reg3_stdVol_1_Date)) #includes residuals from reg 3

Reg4_residStdVol_1 <- feols(abs(logDif) ~ stdVol_1DateResid, data=GSPC)
summary(Reg4_residStdVol_1) #Regression 4 shows that the residuals are also significant predictors of absolute logDif

Reg5_date_residStdVol_1 <- feols(stdVol_1DateResid ~ Date, data=GSPC)
summary(Reg5_date_residStdVol_1) #Regression 5 shows that there is no relationship between the residuals and date

rm(Reg1_stdVol_1_logDif, Reg2_stdVol_1_abslogDif, Reg3_stdVol_1_Date, Reg4_residStdVol_1, Reg5_date_residStdVol_1)

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
  geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "stdVol_DateResid and logDif", y = "stdVolControlled and logDif", x = "Year")

####################################
##ARCH  TESTS - Kevin Kotze tutorial

#res1 - looking for serial dependence i.e. significant lags
res1 <- ac(logDif) #indicates that the model for the mean equation may have no persistence or may be an AR(2/3) or MA(2/3)
summary(res1)

Box.test(logDif, lag=10, type = "Ljung") #checking if serial correlation exists - significant so serial correlation exists
#number of entries allows for up to 18 lags according to the strictest metric

#res2 - indicates degree of persistence of volatility 
res2 <- ac(abs(logDif))

#res3 - again indicates persistence but not as good a measure as absolute because a lot of the entries are lower than 1 and are therefore made negligible by the ^2
res3 <- ac(logDif^2)

#t-test indicates that the mean is not = 0
t.test(logDif)

#demeaning the series - i.e. removing the mean
DlogDif <- logDif-mean(logDif)

#t-test indicates that the DlogDif mean is definitely = 0
t.test(DlogDif)

Box.test(abs(DlogDif), lag = 10, type = 'Ljung') #- There is still serial correlation

#results indicate that ARCH effects are present
archTest(DlogDif,10) # - model the error lags in a normal ts regression (1,2 then 1,2,5,6 then 1,2,5,6,8,9)

DlogDif_1 <- shift(DlogDif,1) 
DlogDif_2 <- shift(DlogDif,2)
DlogDif_5 <- shift(DlogDif,5)
DlogDif_6 <- shift(DlogDif,6)
DlogDif_8 <- shift(DlogDif,8)
DlogDif_9 <- shift(DlogDif,9)

DlogDif_df <- data.frame(DlogDif,DlogDif_1,DlogDif_2,DlogDif_5,DlogDif_6,DlogDif_8,DlogDif_9) #create a dataframe with all the DlogDif lags

rm(res1,res2,res3,DlogDif_1,DlogDif_2,DlogDif_5,DlogDif_6,DlogDif_8,DlogDif_9)

lagCheck0 <- feols(DlogDif ~ DlogDif_1 ,data = DlogDif_df)
lagCheck1 <- feols(DlogDif ~ DlogDif_1 + DlogDif_2,data = DlogDif_df)
lagCheck2 <- feols(DlogDif ~ DlogDif_1 + DlogDif_2 + DlogDif_5 + DlogDif_6,data = DlogDif_df)
lagCheck3 <- feols(DlogDif ~ DlogDif_1 + DlogDif_2 + DlogDif_5 + DlogDif_6 + DlogDif_8 + DlogDif_9,data = DlogDif_df)

summary(lagCheck0)
summary(lagCheck1)
summary(lagCheck2)
summary(lagCheck3)

rm(lagCheck0,lagCheck1, lagCheck2, lagCheck3, logDif)
#these regressions indicate significance on the 2nd, 6th, 8th and 9th lags

ArchTest(DlogDif, lags = 10, demean = TRUE) #also indicates that ARCH effects are present

model1 <- arima(DlogDif, order = c(0,0,1)) #AIC = 50000+ - shouldn't just rely on AIC - use other tests - BIC etc.
model1

model2 <- arima(DlogDif, order = c(1, 0, 0)) #AIC = 50000+
model2

check1 <- ac(residuals(model1, standardize = T)) #delayed residuals still exist for model 1
check2 <- ac(residuals(model2, standardize = T)) #delayed residuals still exist for model 2
#Model 1 and 2 are essentially giving the same result

model3 <- garchFit(DlogDif ~ arma(1,0) + garch(1,1), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.395497
model3
summary(model3) #this is the correct model to use because essentially removes any autocorrelation between its residuals indicating that it has captured all of the relevent information for autocorrelation
check3 <- ac(residuals(model3, standardize = T)) #significant second lag should try AR(2)
#need to include ar1, and variance1 of DlogDif in final data

model3.1 <- garchFit(DlogDif ~ arma(0,2) + garch(1,1), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.395114
model3.1
summary(model3.1)
check3.1 <- ac(residuals(model3.1, standardize = T))

model3.2 <- garchFit(DlogDif ~ arma(2,0) + garch(1,1), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.395102
model3.2
summary(model3.2)
check3.2 <- ac(residuals(model3.2, standardize = T))
#need to include ar1, ar2 and variance1 of DlogDif in final data - this performs slightly better than model3


model4 <- garchFit(DlogDif ~ garch(1, 1), data = DlogDif, cond.dist = "std",  #AIC is 2.36 and suggests a standard AR(1) 
                   trace = F) # the high significance of lag 1 indicates that the AR(1) was better
summary(model4)
check4 <- ac(residuals(model4, standardize = T))

model5 <- garchFit(DlogDif ~ arma(1,0) + garch(1,0), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model5
check5 <- ac(residuals(model5, standardize = T))


model6 <- garchFit(DlogDif ~ garch(1,0), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model6
check6 <- ac(residuals(model6, standardize = T))

par(mfrow = c(1, 1))
plot.ts(residuals(model3.2, standardize = T))

plot.ts(residuals(model3, standardize =T))

rm(model1,model2,model3,model3.1,model3.2,model4,model5,model6,check1,check2,check3,check3.1,check3.2,check4,check5,check6)

#####################################
###Positive vs negative movements

positive_vec <- vector()
negative_vec <- vector()
  
for (i in DlogDif)
{
  if (i >0){
    positive_vec <- c(positive_vec, i)}
  
  else{negative_vec <- c(negative_vec, i)}
  
}

pos_neg_ratio <- mean(negative_vec)/mean(positive_vec)

pos_neg_ratio

pos_neg_transform <- vector()

for (i in DlogDif)
{
  if (i >0){
    pos_neg_transform <- c(pos_neg_transform, 1)}
  
  else{pos_neg_transform <- c(pos_neg_transform, pos_neg_ratio)}
  
}

pos_neg_transform <- shift(pos_neg_transform,1)

rm(negative_vec,positive_vec,i,pos_neg_ratio)
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

model5 <- ugarchfit(spec = x, data = DlogDif) 

model5

check5 <- ac(residuals(model5, standardize = T))

rm(x,model5,check5)

#######################################
##Export data setup

DlogDif_df <- subset(DlogDif_df, select = c(DlogDif,DlogDif_1,DlogDif_2))
DlogDif_df <- DlogDif_df |> mutate(absDlogDif = abs(DlogDif),absDlogDif_1 = abs(DlogDif_1))

output_df <- data.frame(Date = GSPC$Date, DlogDif_df,logDif = GSPC$logDif,
                        logDif_date_resid = GSPC$logDif_date_resid,
                        logDif_date_resid_1 = GSPC$logDif_date_resid_1,
                        blackSwan_SD3_1 = shift(GSPC$blackSwan_SD3,1),
                        blackSwan_SD4_1 = shift(GSPC$blackSwan_SD4,1),
                        blackSwan_SD5_1 = shift(GSPC$blackSwan_SD5,1), 
                        stdVol_1DateResid = GSPC$stdVol_1DateResid,
                        pos_neg_transform)
  
rm(GSPC,DlogDif_df,DlogDif,pos_neg_transform)

funreg <- feols(DlogDif~ i(blackSwan_SD5_1) + i(pos_neg_transform) ,data = output_df)
summary(funreg)

write.csv(output_df,"/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Complete_data/GSPC_features.csv", row.names = FALSE)

rm(list=ls()) #Removes all items in Environment!
graphics.off()

 