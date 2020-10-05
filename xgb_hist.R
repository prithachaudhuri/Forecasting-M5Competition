#### Complicated XGBOOST (0.56) #### 

library(data.table)
library(xgboost)
library(caret)
library(tictoc)

set.seed(21081991)
# setDTthreads(11)
####
free <- function() invisible(gc())
viewdt <- function(dt,n=10){View(head(dt,n))}

#### Pre-processing #### 
## Constants
h <- 28
max_lags <- 366
tr_last <- 1913
fday <- as.IDate("2016-04-25") 

## Load and Merge ##
prices <- fread("m5-forecasting-accuracy/sell_prices.csv")
cal <- fread("m5-forecasting-accuracy/calendar.csv")[,date:=as.IDate(date,format="%Y-%m-%d")]
dt <- fread("m5-forecasting-accuracy/sales_train_validation.csv")
dt[, paste0("d_", (tr_last+1):(tr_last+2*h)) := NA_real_]

# wide to long
dt <- melt(dt,
           measure.vars = patterns("^d_"),
           variable.name = "d",
           value.name = "sales")

# merge to calendar
dt[cal,`:=`(date=i.date,
            wm_yr_wk=i.wm_yr_wk,
            event_name_1=i.event_name_1,
            snap_CA=i.snap_CA,
            snap_TX=i.snap_TX,
            snap_WI=i.snap_WI),on="d"]

# merge to prices
dt[prices, sell_price := i.sell_price, on=c("store_id", "item_id", "wm_yr_wk")]
# table(dt$event_name_1)

# encode categorical variables
free()
cat.enc <- dt[,c("id","date","item_id","dept_id","cat_id","state_id","store_id","sales","event_name_1")]

cat.enc[, item_id_tgt := mean(sales,na.rm=T),by="item_id"]
cat.enc[, dept_id_tgt := mean(sales,na.rm=T),by="dept_id"]
cat.enc[, cat_id_tgt := mean(sales,na.rm=T),by="cat_id"]
cat.enc[, store_id_tgt := mean(sales,na.rm=T),by="store_id"]
cat.enc[, state_id_tgt := mean(sales,na.rm=T),by="state_id"]
cat.enc[, event_name_tgt := mean(sales,na.rm=T),by="event_name_1"]

dt[cat.enc,`:=`(item_id_tgt=i.item_id_tgt,dept_id_tgt=i.dept_id_tgt,
                cat_id_tgt=i.cat_id_tgt,store_id_tgt=i.store_id_tgt,
                state_id_tgt=i.state_id_tgt,event_name_tgt=i.event_name_tgt),
   on = c("id","date")]

rm(cat.enc)
free()

dt[, `:=`(wday = wday(date),
          mday = mday(date),
          week = week(date),
          month = month(date),
          quarter = quarter(date),
          year = year(date))]

## separate test and train set
test <- dt[date > fday-max_lags] # keep all date from a before real test date
train <- dt[date < fday]
rm(dt)
free()

# create some new variables
train[,(paste0("lag_",c(7,28))) := shift(.SD,c(7,28)), .SDcols = "sales", by = "id"]
train[,(paste0("rmean7_",c(7,28))) := frollmean(lag_7, c(7,28), na.rm = TRUE), by = "id"]
train[,(paste0("rmean28_",c(7,28))) := frollmean(lag_28, c(7,28), na.rm = TRUE), by = "id"]

rm(cal,prices)
free()
train <- train[complete.cases(train)]

## Run XGBoost ##
y <- train$sales
train <- data.matrix(train[,c("id","item_id","dept_id","cat_id","state_id",
                              "event_name_1","date","sales","d","wm_yr_wk","store_id") := NULL])
free()

## Simple XGBoost ##
p <- list(eta = 0.075,
          objective = "count:poisson",
          eval_metric = "rmse",
          lambda = 0.1,
          colsample_bytree = 0.77,
          nthread=11) ### ChANGE nthreads IN KAGGLE

tic()
bst.encvars <- xgboost(data=train,label=y,params=p,verbose=T,nrounds=2000,early_stopping_rounds=10,
                       tree_method="hist")
toc()

xgb.plot.importance(xgb.importance(model=bst.encvars), top_n = 20,measure="Gain")
save(bst.encvars,file="bst.encvars.Rdata")
rm(train)
free()

### Test data
for (day in as.list(seq(fday,length.out=2*h, by="day"))){
  cat(as.character(day), " ")
  tst <- test[date >= day - max_lags & date <= day]
  tst[,(paste0("lag_", c(7,28))) := shift(.SD,c(7,28)), .SDcols = "sales", by = "id"]
  tst[,(paste0("rmean7_",c(7,28))) := frollmean(lag_7, c(7,28), na.rm = TRUE), by = "id"]
  tst[,(paste0("rmean28_",c(7,28))) := frollmean(lag_28, c(7,28), na.rm = TRUE), by = "id"]
  
  
  tst <- data.matrix(tst[date == day][,c("id","item_id","dept_id","cat_id","state_id",
                                         "event_name_1","date","sales","d","wm_yr_wk","store_id") := NULL])
  test[date == day, sales := 1.02*predict(bst.encvars, tst)] # magic multiplier by kyakovlev
}


submit <- test[date>= fday]
submit[date >= fday+h, id := sub("validation", "evaluation", id)]
submit[, d := paste0("F", 1:28), by = id]
submit <- submit[,c("id","d","sales"),with=F]
submit[, dcast(.SD, id ~ d, value.var = "sales")][,fwrite(.SD, "submit/sub_dt_xgb_enc.csv")]

























