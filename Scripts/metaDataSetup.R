rm(list=ls()) #Removes all items in Environment!
graphics.off()

source('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Scripts/std_fin-ts_data_setup.R')

BTC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/BTC.csv" , header = T) #this should load from a sql database rather
Oil <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/OIL.csv" , header = T) #this should load from a sql database rather
Nasdaq <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/NASDAQ_comp.csv" , header = T) #this should load from a sql database rather
USDX <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/USDX.csv" , header = T) #this should load from a sql database rather
VIX <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/VIX.csv" , header = T) #this should load from a sql database rather
SSE <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/SSE_comp.csv" , header = T) #this should load from a sql database rather
GSPC <- read.csv("/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/GSPC_features.csv", header = T)

output_df <- data.frame(Date = as.Date(GSPC$Date, format ="%Y-%m-%d" ))
rm(GSPC)

BTC <- stdDataSetup(BTC)
Oil <- stdDataSetup(Oil)
Nasdaq <- stdDataSetup(Nasdaq)
USDX <- stdDataSetup(USDX)
VIX <- stdDataSetup(VIX)
SSE <- stdDataSetup(SSE)

BTC <- get_ld_dr(BTC,'BTC')
Oil <- get_ld_dr(Oil,'Oil')
Nasdaq <- get_ld_dr(Nasdaq,'Nasdaq')
USDX <- get_ld_dr(USDX,'USDX')
VIX <- get_ld_dr(VIX,'VIX')
SSE <- get_ld_dr(SSE,'SSE')

output_df <- left_join(output_df,BTC)
rm(BTC)
output_df <- left_join(output_df,Nasdaq)
rm(Nasdaq)
output_df <- left_join(output_df,Oil)
rm(Oil)
output_df <- left_join(output_df,SSE)
rm(SSE)
output_df <- left_join(output_df,USDX)
rm(USDX)
output_df <- left_join(output_df,VIX)
rm(VIX)

write.csv(output_df,"/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/MetaData/metadata.csv", row.names = FALSE)




