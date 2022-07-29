pacman::p_load(tidyverse,tseries,lubridate, dplyr,fixest)

library(tidyverse)
library(tseries)
library(dplyr)
library(fixest)

#shift function
shift <- function(x, n=1){
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

#logdiff function
logdiff <- function(x){
  ld <- 100*(log(x) - log(shift(x,1)))
}

#std setup function
stdDataSetup <- function(fin_ts_df,volFilter = F,omit_na = T){

  if (volFilter ==T) {
    fin_ts_df <- fin_ts_df |>  dplyr::filter(Volume > 0)}

  #if (omit_na == T) fin_ts_df <-  na.omit(fin_ts_df)

  if (omit_na == T) fin_ts_df <- fin_ts_df[complete.cases(fin_ts_df$Close),]

  fin_ts_df <- fin_ts_df %>% dplyr::mutate(Date = as.Date(Date, format ="%Y-%m-%d" )) %>%
    arrange(desc(Date)) %>%
    mutate(Close_1 = shift(Close,1)) %>%
    mutate(DPC = ret(Close)) %>% mutate(logDif = logdiff(Close)) |>
    mutate(logDif_1 = shift(logDif,1)) |>
    mutate(stdVol = std(Volume)) %>% mutate(stdVol_1 = shift(stdVol,1)) %>%
    arrange(Date)

  if (omit_na == T) fin_ts_df <- fin_ts_df[complete.cases(fin_ts_df$logDif),]

    fin_ts_df <- fin_ts_df[-1,]

    start <- c(format(first(fin_ts_df$Date),"%Y"),format(first(fin_ts_df$Date),"%d"))
    freq <- abs(round(length(fin_ts_df$Date)/(as.integer(format(last(fin_ts_df$Date),"%Y")) - as.integer(format(first(fin_ts_df$Date),"%Y")))))
    fin_ts_df <- fin_ts_df |> arrange(desc(Date))

    reg1 <- feols(logDif ~ Date, data = fin_ts_df)
    fin_ts_df <- fin_ts_df %>% mutate(logDif_date_resid = resid(reg1)) |>
      mutate(logDif_date_resid_1 = shift(logDif_date_resid,1)) |>
        mutate(logDif_date_resid_2 = shift(logDif_date_resid_1,1))

    fin_ts_df
}

#get ld_1 & dr_1 function
get_ld_dr <- function(fin_ts_df,ticker){

  ldname <- paste(ticker,'_ld_1',sep ='')
  drname <- paste(ticker,'_dr_1',sep ='')

  fin_ts_df <- fin_ts_df |> mutate( ld_1 = logDif_1, dr_1 = logDif_date_resid_1) |>
    select(Date,ld_1,dr_1)

  names(fin_ts_df)[names(fin_ts_df) == "ld_1"] <- ldname
  names(fin_ts_df)[names(fin_ts_df) == "dr_1"] <- drname


  fin_ts_df

}


blackSwan <- function(fin_ts_df, numSD){

  bsname <- paste('blackSwan_SD',as.character(numSD) ,sep ='')

  stdDev = sd(fin_ts_df$logDif_date_resid)

  fin_ts_df <- fin_ts_df |> mutate(blackSwan = ifelse(abs(logDif_date_resid) > numSD*stdDev,1,0))

  names(fin_ts_df)[names(fin_ts_df) == "blackSwan"] <- bsname

  fin_ts_df
}
