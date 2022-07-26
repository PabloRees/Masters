

#Volume analysis
volumeAnalysis <- function(GSPC){

    #stdVol and Daily%Change plot
    ############################################ Second plot for writeup
    figure5 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = stdVol_1), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 0.5, color = "darkgreen")+ # similarly for points
        scale_x_date(date_labels = "'%y", date_breaks = "year") +
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
        scale_x_date(date_labels = "'%y", date_breaks = "year") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "logDif over time", y = "logDif", x = "Year")

    figure4

    figure6 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        scale_x_date(date_labels = "'%y", date_breaks = "year") +
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
        scale_x_date(date_labels = "'%y", date_breaks = "year") +
        # make x axis labels
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        # rotate x axis labels
        labs(title = "stdVol_DateResid ", y = "DateResid of StdVol and stdVol", x = "Year")

    figure7

    figure8 <- GSPC %>% ggplot() + # creates the 'canvas'
        theme_bw() + # choose on of many existing themes
        geom_line(aes(x = Date, y = abs(logDif)), size = 0.1, alpha = 1, color = "firebrick4") + # creates the line on the canvas with aes() coordinates
        geom_line(aes(x = Date, y = stdVol_1DateResid), size = 0.1, alpha = 1, color = "darkgreen") + # creates the line on the canvas with aes() coordinates
        scale_x_date(date_labels = "'%y", date_breaks = "year") +
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

    model3.2 <- garchFit(DlogDif ~ arma(2,0) + garch(1,1), data = DlogDif, trace = F) #indicates an AR(2) model might be appropriate, AIC is 2.395102
    model3.2
    summary(model3.2)
    check3.2 <- ac(residuals(model3.2, standardize = T))
    #need to include ar1, ar2 and variance1 of DlogDif in final data - this performs slightly better than model3


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

    par(mfrow = c(1, 1))
    plot.ts(residuals(model3.2, standardize = T))

    plot.ts(residuals(model3, standardize =T))

    returnList <-  list(mod3resid , mod4resid , mod5resid , mod6resid)

    rm(model1,model2,model3,model3.1,model3.2,model4,model5,model6,check1,check2,check3,check3.1,check3.2,check4,check5,check6)

    returnList

}