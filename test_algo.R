linreg = function(x,y){ lm1 = lm(y ~ x)
                     flm1 = fitted(lm1)
                     return( flm1)
                      }

rmse = function(fit, raw){  rmse = sqrt(mean((fit-raw)^2))
                            return(rmse)
                          }

minmax = function(y){ min_max = (y - min(y))/(max(y)-min(y))
                    return(min_max)
                      
                    }


sliding_window = function(x,y,n=100){
  
                          seqofnumber = seq(1, length(y), length.out = n)
                          
                          setClass(Class="RMSE",
                                   representation(
                                     y = "numeric",
                                     seqofnumber="numeric",
                                     rmse="numeric"
                                   )
                          )
                          
                          rmse_res = c()
                          for( i in seqofnumber) {
                            flm1 = linreg(x[1:i], y[1:i])
                            rmse1 = rmse(flm1, y[1:i])
                            rmse_res=append(rmse_res,rmse1)
                          }
                          
                          if(mean(rmse_res) < 0.01){
                            rmse_res = rmse_res
                          } else{
                            rmse_res = minmax(rmse_res)
                          }
                          
                          return(new("RMSE",
                                     y=y,
                                     seqofnumber=seqofnumber,
                                     rmse=rmse_res))
                          }


find_intersect = function(y, rmse){
  thediff = diff(sign(y - rmse))
  intersect = which(thediff !=0)
  return(intersect)
}

viz_error = function(x,y, raw, title = "no transfo"){
                                error = sliding_window(x, y, n = length(y))
                                intersect = find_intersect(raw, error@rmse)
                                print(intersect)
                                plot(error@seqofnumber, minmax(y), type ='l', ylim = c(0,1),main = paste0(title, " transformed; crossing at : ", intersect))
                                lines(x, minmax(raw), col = 'blue')
                                lines(error@seqofnumber, error@rmse, col = 'red')
                                abline(v=x[intersect], col = 'darkgreen', lty = 2)
                                abline(h=raw[intersect], col = 'darkgreen', lty = 2)
                                legend("topright", legend=c("transformed signal", "RMSE on transformed", "Raw signal"),
                                       col=c("black", "red", "blue"), lty=1:2, cex=0.8)
                                
                              }


# s

# Error lin

X = seq(1,1000)
A = 1
B= 0.08
E = A * exp(-B*X)
plot(X, E)




par(mfrow= c(2,2))
viz_error(X, E, E)

viz_error(X,log(E), E, title = "log")

viz_error(X,-1/exp(E), E, title = "1/exp")

viz_error(X,sqrt(E), E, title = "root")

dev.off()






