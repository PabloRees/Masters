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

##
## DATA SETUP

MasterGSPC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC.csv" , header = TRUE)

WorkingGSPC <- MasterGSPC %>% dplyr::mutate(logClose = log(Close), Date = as.Date(Date, format ="%Y-%m-%d" )) %>%
  dplyr::filter(Date > as.Date('1949-12-30'))
WorkingGSPC <- WorkingGSPC[order(WorkingGSPC$Date, decreasing = TRUE),]
WorkingGSPC <- WorkingGSPC%>%mutate(daysSince = Date - as.Date('1949-12-30'),Close_1 = shift(WorkingGSPC$Close,1), stdVol = (WorkingGSPC$Volume - mean(WorkingGSPC$Volume))/sd(WorkingGSPC$Volume))

WorkingGSPC <- WorkingGSPC%>%mutate(stdVol_1 = shift(WorkingGSPC$stdVol,1),Volume_1 = shift(WorkingGSPC$Volume,1), volPC = -100*(diff(WorkingGSPC$Volume))/Volume_1,volPC_1 = shift(volPC,1))

volDiff = diff(WorkingGSPC$Volume)
Diff <- -diff(WorkingGSPC$Close)
WorkingGSPC <- head(WorkingGSPC,-1)
WorkingGSPC <- WorkingGSPC%>%mutate(Diff = Diff, dailyPercentageChange = 100*(Diff/Close_1),volDiff = volDiff)

WorkingGSPC <- WorkingGSPC[order(WorkingGSPC$Date, decreasing = FALSE),]

start <- c(format(WorkingGSPC$Date[[1]],"%Y"),format(WorkingGSPC$Date[[1]],"%d"))
freq <- round(length(WorkingGSPC$Date)/(as.integer(format(tail(WorkingGSPC$Date,1),"%Y")) - as.integer(format(WorkingGSPC$Date[[1]],"%Y"))))

#####################################
####Volume analysis

#stdVol and Daily%Change plot
WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = stdVol_1), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = abs(dailyPercentageChange)), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Lagged Standardized Volume and % Daily Change", y = "stdVol_1 and % Daily Change", x = "Year")

#Daily%change plot
WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = dailyPercentageChange), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Daily Percentage Change", y = "LogPrice", x = "Year")

WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = (dailyPercentageChange)^2), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Daily Percentage Change Squared ", y = "LogPrice", x = "Year")

WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = abs(dailyPercentageChange)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = (stdVol)), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "Absolute Daily Percentage Change and stdVol", y = "Absolute DPC and stdVol", x = "Year")


stdVol_1Reg <- feols(dailyPercentageChange ~ stdVol_1, data = WorkingGSPC) #shows insignificance of volume as a predictor of DCP 
summary(stdVol_1Reg)

stdVol_1_absDPCReg <- feols(abs(dailyPercentageChange) ~ stdVol_1, data = WorkingGSPC) #shows high significance of volume as a predictor of absolute DCP
summary(stdVol_1_absDPCReg)

stdVol_1Date_reg <- feols(stdVol_1~Date, data=WorkingGSPC) #shows that stdVol_1 is not stationary over time
summary(stdVol_1Date_reg)
  

WorkingGSPC <- WorkingGSPC %>% mutate(stdVol_1DateResid = resid(stdVol_1Date_reg))

residStdVol_1reg <- feols(abs(dailyPercentageChange) ~ stdVol_1DateResid, data=WorkingGSPC)
summary(residStdVol_1reg)

date_residStdVol_1reg <- feols(stdVol_1DateResid ~ Date, data=WorkingGSPC)
summary(date_residStdVol_1reg)

WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = stdVol), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "stdVol_DateResid ", y = "DateResid of StdVol and stdVol", x = "Year")

WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = abs(dailyPercentageChange)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "stdVol_DateResid and DPC", y = "stdVolControlled and DPC", x = "Year")

cov(abs(WorkingGSPC$dailyPercentageChange),WorkingGSPC$stdVol_1DateResid)

####################################
##ARCH  TESTS - Kevin Kotze tutorial

WorkingGSPC <- WorkingGSPC[order(WorkingGSPC$Date, decreasing = FALSE),]
DPC <-  ts(WorkingGSPC$dailyPercentageChange , start=start , freq = freq)
stdVol <-  ts(WorkingGSPC$stdVol ,start=start, fre=freq)

#res1 - looking for serial dependence i.e. significant lags
res1 <- ac(DPC) #indicates that the model for the mean equation may have no persistence or may be an AR(2/3) or MA(2/3)
summary(res1)

Box.test(DPC, lag=10, type = "Ljung") #checking if serial correlation exists - significant so serial exists

#res2 - indicates degree of persistence of volatility 
res2 <- ac(abs(DPC))

#res3 - not sure what we're looking for here
res3 <- ac(DPC^2)

#t-test indicates that the mean is not = 0
t.test(DPC)

#demeaning the series - i.e. removing the mean
DDPC <- DPC-mean(DPC)

#t-test indicates that the mean is definitely = 0
t.test(DDPC)

Box.test(abs(DDPC), lag = 10, type = 'Ljung') #- There is serial correlation

#results indicate that ARCH effects are present
archTest(DDPC,10) # - model the error lags in a normal ts regression (1,2 then 1,2,5,6 then 1,2,5,6,8,9)

DDPC_1 <- shift(DDPC,1) 
DDPC_2 <- shift(DDPC,2)
DDPC_5 <- shift(DDPC,5)
DDPC_6 <- shift(DDPC,6)
DDPC_8 <- shift(DDPC,8)
DDPC_9 <- shift(DDPC,9)

#DDPC_0 <- data.frame(DDPC)
#DDPC_1 <- data.frame(DDPC_1) 
#DDPC_2 <- data.frame(DDPC_2)
#DDPC_5 <- data.frame(DDPC_5)
#DDPC_6 <- data.frame(DDPC_6)
#DDPC_8 <- data.frame(DDPC_8)
#DDPC_9 <- data.frame(DDPC_9)

DDPC_df <- data.frame(DDPC,DDPC_1,DDPC_2,DDPC_5,DDPC_6,DDPC_8,DDPC_9)

lagCheck0 <- feols(DDPC ~ DDPC_1 ,data = DDPC_df)
lagCheck1 <- feols(DDPC ~ DDPC_1 + DDPC_2,data = DDPC_df)
lagCheck2 <- feols(DDPC ~ DDPC_1 + DDPC_2 + DDPC_5 + DDPC_6,data = DDPC_df)
lagCheck3 <- feols(DDPC ~ DDPC_1 + DDPC_2 + DDPC_5 + DDPC_6 + DDPC_8 + DDPC_9,data = DDPC_df)

summary(lagCheck0)
summary(lagCheck1)
summary(lagCheck2)
summary(lagCheck3)

ArchTest(DDPC, lags = 10, demean = TRUE) #also indicates that ARCH effects are present

model1 <- arima(DDPC, order = c(0,0,1)) #AIC = 50000+ - shouldn't just rely on AIC - use other tests - BIC etc.
summary(model1)  
  
model2 <- arima(DDPC, order = c(1, 0, 0)) #AIC = 50000+
model2
summary(model2)

check <- ac(residuals(model1, standardize = T))
check <- ac(residuals(model2, standardize = T))

#Model 1 and 2 are essentially giving the same result
model3 <- garchFit(DDPC ~ arma(1,0) + garch(1,1), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model3
summary(model3) #this is the correct model to use because it kills all the residuals

model4 <- garchFit(DDPC ~ garch(1, 1), data = DDPC, cond.dist = "std",  #AIC is 2.35 and suggests a standard AR(1) 
               trace = F) # the high significance of lag 1 indicates that the AR(1) was better
summary(model4)

model6 <- garchFit(DDPC ~ arma(1,0) + garch(1,0), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model6

model7 <- garchFit(DDPC ~ garch(1,0), data = DDPC, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
model7

check1 <- ac(residuals(model4, standardize = T))

check2 <- ac(residuals(model3, standardize = T))

check3 <- ac((residuals(model3, standardize = T)^2)^0.5)

check4 <- ac((residuals(model3, standardize = T)^2))

check5 <- ac(abs(residuals(model3, standardize = T)))

check7 <- ac(residuals(model6, standardize = T))

check8 <- ac(residuals(model7, standardize = T))

par(mfrow = c(1, 1))
plot.ts(residuals(model4, standardize = T))

plot.ts(residuals(model3, standardize =T))

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

check6 <- ac(residuals(model5, standardize = T))

######################################
##Volume Auto Correlation and ARCH Tests

WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = volDiff), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "StdVol ", y = "StdVol", x = "Year")

WorkingGSPC %>% ggplot() + # creates the 'canvas'
  theme_bw() + # choose on of many existing themes
  geom_line(aes(x = Date, y = volPC), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
  scale_x_date(date_labels = "'%y", date_breaks = "year") +
  # make x axis labels
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  # rotate x axis labels
  labs(title = "StdVol ", y = "StdVol", x = "Year")

mean(WorkingGSPC$volPC)

acf_volPC <- stats::acf(
  x = WorkingGSPC$volPC,
  plot = T,
  type = "correlation"
)


pacf_vol <- stats::acf(
  x = WorkingGSPC$volPC,
  plot = T,
  type = "partial") # we want the PACF option

library(urca)
ADF1_vol <- ur.df(
  y= WorkingGSPC$volPC, #vector
  type = "trend",
  lags = 3,
  selectlags = "AIC")

summary(ADF1_vol) #yields 1 significant lag and a p-score less than 0,05 which suggests stationarity

WorkingGSPC <- WorkingGSPC[order(WorkingGSPC$Date, decreasing = TRUE),]

volPC_2 = shift(WorkingGSPC$volPC_1,1)
volPC_3 = shift(volPC_2,1)
volPC_4 = shift(volPC_2,2)
volPC_5 = shift(volPC_2,3)

WorkingGSPC <- WorkingGSPC[order(WorkingGSPC$Date, decreasing = FALSE),]

GSPCVols <- WorkingGSPC %>% dplyr::select(Date,Volume, Volume_1, volPC, volPC_1) %>%
  data.frame(volPC_2,volPC_3,volPC_4,volPC_5)


volPC_AR1 <- feols(volPC ~ volPC_1 + volPC_2 + volPC_3+volPC_4+ volPC_5, data=GSPCVols)
summary(volPC_AR1)

volPC_AR2 <- feols(volPC ~ volPC_1 + volPC_2 + volPC_3, data=GSPCVols)
summary(volPC_AR2)

#These suggest AR(1) models

####################################
##ARCH  TESTS - Kevin Kotze tutorial
WorkingGSPC <- WorkingGSPC[order(WorkingGSPC$Date, decreasing = TRUE),]
start <- c(format(WorkingGSPC$Date[[1]],"%Y"),format(WorkingGSPC$Date[[1]],"%d"))
freq <- round(length(WorkingGSPC$Date)/(as.integer(format(tail(WorkingGSPC$Date,1),"%Y")) - as.integer(format(WorkingGSPC$Date[[1]],"%Y"))))

vol <-  ts(WorkingGSPC$volPC , start=start , freq = freq)

#res1 - looking for serial dependence i.e. significant lags
res1 <- ac(vol) #indicates that the model for the mean equation may be AR(4) or MA(4)
summary(res1)

Box.test(vol, lag=10, type = "Ljung") #checking if serial correlation exists - significant so serial exists

#res2 - indicates degree of persistence of volatility - I think this indicates that this should be an arch model - 
res2 <- ac(abs(vol))

#res3 - not sure what we're looking for here
res3 <- ac(vol^2)

#t-test indicates that the mean is not = 0
t.test(vol)

#demeaning the series - i.e. removing the mean
at <- vol-mean(vol)

#t-test indicates that the mean is definitely = 0
t.test(at)

Box.test(abs(at), lag = 10, type = 'Ljung') #- There is serial correlation

#results indicate that ARCH effects are not present
archTest(at,10) # - 

ArchTest(vol, lags = 10, demean = TRUE) #also indicates that ARCH effects are not present

#######################################
##Auto Correlation and Dickey Fuller Tests

acf <- stats::acf(
  x = SnP,
  plot = T,
  type = "correlation"
)


pacf <- stats::acf(
  x = SnP,
  plot = T,
  type = "partial") # we want the PACF option

library(urca)
ADF1 <- ur.df(
  y= SnP, #vector
  type = "trend",
  lags = 3,
  selectlags = "AIC")

summary(ADF1) #yields 1 significant lag and a p-score less than 0,05 which suggests stationarity

#############################
##ARCH Tests - Bookdown.org

CTS <- ts(SnP)
hist(CTS, main="", breaks=50, freq=FALSE, col="darkgreen")

dpcARCH_test <- FinTS::ArchTest(CTS, lags = 1, demean = TRUE)
dpcARCH_test

CGARCH_test <- tseries::garch(CTS)
summary(CGARCH_test)

CARCH_test2 <- garch(CTS,c(0,1))
summary(CARCH_test2)

hhat <- ts(2*CARCH_test2$fitted.values[-1,1]^2)
plot.ts(hhat)


