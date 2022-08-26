source('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Scripts/std_fin-ts_data_setup.R')

everysecond <- function(x){
  x <- sort(unique(x))
  x[seq(2, length(x), 2)] <- ""
  x
}

#Volume analysis
volumeAnalysis <- function(GSPC){

    #stdVol and Daily%Change plot
    ############################################ Second plot for writeup
    figure5 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = stdVol_1), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
        scale_x_date(date_labels = "'%y", date_breaks = "2 years") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "Lagged Standardized Volume and logDif", y = "stdVol_1 and absolute logDif", x = "Year")

    figure5

    #Daily%change plot
    ############################################ First plot for writeup
    figure4 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = logDif), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        scale_x_date(date_labels = "'%y", date_breaks = "2 years") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "logDif over time", y = "logDif", x = "Year")

    figure4

    figure6 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        scale_x_date(date_labels = "'%y", date_breaks = "2 years") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "Absolute logDif over time", y = "Absolute logDif", x = "Year")

    figure6

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

    figure7 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        geom_line(aes(x = Date, y = stdVol), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
        scale_x_date(date_labels = "'%y", date_breaks = "2 years") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "stdVol_DateResid ", y = "DateResid of StdVol and stdVol", x = "Year")

    figure7

    figure8 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
        scale_x_date(date_labels = "'%y", date_breaks = "2 years") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "stdVol_DateResid and absolute logDif", y = "stdVolControlled and absolute logDif", x = "Year")

    figureList <- list(figure4, figure5, figure6, figure7, figure8)

    figureList
}

ARCHTests <- function(logDif){

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
    mod3resid <- residuals(model3, standardize = T)
    check3 <- ac(mod3resid) #significant second lag should try AR(2)
    #need to include ar1, and variance1 of DlogDif in final data

    model3.1 <- garchFit(DlogDif ~ arma(0,2) + garch(1,1), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.395114
    model3.1
    summary(model3.1)
    check3.1 <- ac(residuals(model3.1, standardize = T))

    model4 <- garchFit(DlogDif ~ garch(1, 1), data = DlogDif, cond.dist = "std",  #AIC is 2.36 and suggests a standard AR(1)
                       trace = F) # the high significance of lag 1 indicates that the AR(1) was better
    summary(model4)
    mod4resid <- residuals(model4, standardize = T)
    check4 <- ac(mod4resid)

    model5 <- garchFit(DlogDif ~ arma(1,0) + garch(1,0), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
    model5
    mod5resid <- residuals(model5, standardize = T)
    check5 <- ac(mod5resid)


    model6 <- garchFit(DlogDif ~ garch(1,0), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.402377
    model6
    mod6resid <- residuals(model6, standardize = T)
    check6 <- ac(mod6resid)

    model8 <- garchFit(DlogDif ~ arma(2,0) + garch(1,1), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.395102
    model8
    summary(model8)
    mod8resid <- residuals(model8, standardize = T)

    check8 <- ac(residuals(model8, standardize = T))
    #need to include ar1, ar2 and variance1 of DlogDif in final data - this performs slightly better than model3

    par(mfrow = c(1, 1))
    plot.ts(residuals(model8, standardize = T))

    plot.ts(residuals(model3, standardize =T))

    ##EGARCH

    pacman::p_load(rugarch, FinTS, zoo, e1071)
    library(rugarch)
    library(FinTS)
    library(zoo)
    library(e1071)

    x <-  ugarchspec(variance.model =
                         list(model="eGARCH", garchOrder=c(1,1)),mean.model =
                         list(armaOrder=c(0,0)))

    model7 <- ugarchfit(spec = x, data = DlogDif)
    model7
    mod7resid <- residuals(model7, standardize = T)
    check7 <- ac(mod7resid)

    returnList <-  list(mod3resid , mod4resid , mod5resid , mod6resid, mod7resid, mod8resid, model3, model4, model5, model6, model7, model8,DlogDif_df)

    rm(model1,model2,model3,model3.1,model8,model4,model5,model6,check1,check2,check3,check3.1,check8,check4,check5,check6,x,model7,check7)

    returnList

}

catTSAnal <- function(GSPC){


    GSPC <- GSPC %>% mutate(Weekday = weekdays(as.Date(Date))) %>%
        mutate(Month = months(as.Date(Date))) %>%
        mutate(Year = format(as.Date(Date),"%Y")) %>%
        mutate(Monthday = format(as.Date(Date),"%d"))

    logDif1 <- GSPC %>% dplyr::select(logDif)


    change_pacf <- stats::acf(
        x = logDif1,
        plot = F,
        type = "partial") # we want the PACF option

    change_pacf

    logDifVector <- logDif1%>% pull(logDif)

    change_ADF1 <- ur.df(
        y= logDifVector, #vector
        type = "trend",
        lags = 3,
        selectlags = "AIC")

    summary(change_ADF1) #yields 1 significant lag and a p-score less than 0,05 which suggests stationarity


    LD_1 <- shift(logDifVector, 1)
    LD_2 <- shift(logDifVector, 2)
    LD_6 <- shift(logDifVector, 6)
    LD_8 <- shift(logDifVector, 8)
    LD_9 <- shift(logDifVector, 9)
    LD_12 <- shift(logDifVector, 12)
    LD_15 <- shift(logDifVector, 15)
    LD_16 <- shift(logDifVector, 16)
    LD_18 <- shift(logDifVector, 18)
    LD_26 <- shift(logDifVector, 26)
    LD_27 <- shift(logDifVector, 27)
    LD_29 <- shift(logDifVector, 29)
    LD_34 <- shift(logDifVector, 34)


    stdVol <-  (GSPC$Volume - mean(GSPC$Volume))/sd(GSPC$Volume)
    stdVol
    stdVol_1 <- shift(stdVol, 1)
    stdVol_2 <- shift(stdVol, 2)
    stdVol_3 <- shift(stdVol, 3)

    logDifAR_df <- GSPC %>% dplyr::select(Date, Weekday, Monthday, Month, Year) %>%
        data.frame(logDifVector, stdVol_1, stdVol_2, stdVol_3, LD_1, LD_2, LD_6, LD_8, LD_9, LD_12, LD_15, LD_16, LD_18, LD_26,LD_27,LD_29, LD_34)

    library(fixest)


    #         regression (dependent ~ independent + independent, data = dataframe)
    logDifAR_full <- feols(logDifVector ~ LD_2 + LD_6 + LD_8 + LD_12 + LD_15 + LD_16 + LD_18 + LD_26 + LD_27 + LD_29 + LD_34, data = logDifAR_df) # Adding an explanatory continuous variable

    summary(logDifAR_full)

    #         dropping LD's with * or ** significance - Adj R2 got worse
    DailyChangeAR_highSig <- feols(logDifVector ~   LD_12 + LD_15 + LD_16 + LD_26  + LD_34, data = logDifAR_df) # Adding an explanatory continuous variable

    summary(DailyChangeAR_highSig)

    vol_reg <-  feols(logDifVector ~  stdVol_1 ,data=logDifAR_df)
    summary(vol_reg)

    dayRegression <- feols(logDif ~ i(Weekday) + i(Monthday) + i(Month) + i(Year), data = GSPC)
    summary(dayRegression)

    logDifAR_df <- logDifAR_df %>% mutate(September = ifelse(Month == 'September',1,0), Monday = ifelse(Weekday == 'Monday',1,0))

    logDifAllSigModel  <- feols(logDifVector ~ stdVol_1  + LD_2 + LD_6 + LD_8 + LD_12 + LD_15 + LD_16 + LD_18 + LD_26 + LD_27 + LD_29 + LD_34  + i(September) + i(Monday) , data = logDifAR_df) # Adding an explanatory continuous variable

    summary(logDifAllSigModel)

    septMond_reg <- feols(logDifVector ~ i(September) + i(Monday), data = logDifAR_df)

    returnList <-  list(logDifAR_full,DailyChangeAR_highSig,vol_reg,dayRegression,logDifAllSigModel,septMond_reg)

}

plotPACF <- function(GSPC){

    logDif <- GSPC %>% dplyr::select(logDif)


    change_pacf <- stats::acf(
        x = logDif,
        plot = T,
        type = "partial") # we want the PACF option
}