##### Simple combination of univariate models #####

#### libraries ####
library(data.table)
library(forecast)
library(doParallel)
library(forecTheta)

####
free <- function() invisible(gc())
viewdt <- function(dt,n=10){View(head(dt,n))}

#### data ####
dt <- fread("m5-forecasting-accuracy/sales_train_validation.csv")
dt[,c("item_id","dept_id","cat_id","store_id","state_id") := NULL]

# Fit scum models
id <- unique(dt$id)

t1 <- Sys.time()
cl <- makeCluster(11, outfile="")
registerDoParallel(cl)
scum.fcs <- foreach(i = 1:length(id),.inorder=FALSE,.combine='rbind',
                    .packages=c('data.table','forecast','forecTheta', 'smooth'))%dopar%{
                      
                      y <- ts(as.numeric(dt[id==id[i],-c("id")]),frequency=1)
                      
                      ets<- as.vector(forecast(ets(y),h=56)$mean)
                      ces <- as.vector(forecast(auto.ces(as.vector(y),interval="p"),h=56)$mean)
                      ses <-  as.vector(ses(y,h=56)$mean)
                      arima <- as.vector(forecast(auto.arima(y),h=56)$mean)
                      dotm <- as.vector(dotm(y,h=56)$mean)
                      # Median combiation
                      comb <- sapply(1:56, function(x) median(c(ets[x],ces[x],ses[x],arima[x],dotm[x])))
                      c(id[i],comb)
                      }
stopCluster(cl)
Sys.time()-t1 # 1.8 hours

scum <- data.table(scum.fcs)
colnames(scum) <- c("id",paste(paste0("F", 1:56)))
rownames(scum) <- NULL

scum <- melt(scum,
             measure.vars = paste(paste0("F", 1:56)),
             variable.name = "d",
             value.name = "sales")
scum[,days := as.numeric(gsub("F","",d))]
scum[days >= 29 ,id := sub("validation", "evaluation", id)]
scum[days >= 29 ,d := paste0("F",days-28)]

scum[,dcast(.SD, id ~ d, value.var = "sales")
       ][,fwrite(.SD,"submit/scum.csv")]


## All zeros
scum[, sales:= 0
  ][,dcast(.SD, id ~ d, value.var = "sales")
     ][,fwrite(.SD,"submit/zer.csv")]














